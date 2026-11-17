"""Aggregate input diagnostics on frozen train+validation; separate historical full-data check."""
import argparse
import json
from pathlib import Path
from urllib.parse import urlsplit

from dataset import load_dataset
from frozen_splits import load_manifest
from preprocessing import normalize_frame
from provenance import sha256


def parse_input(url):
    # A stripped authority must not be interpreted as a relative path or a scheme.
    return urlsplit(url if '://' in url or url.startswith('//') else '//' + url)


def parsed_features(url):
    try:
        parsed = parse_input(url)
        parsed.hostname  # validate bracket syntax; never resolve or fetch
        return bool(parsed.path), bool(parsed.query), False
    except ValueError:
        return False, False, True


def percent(count, denominator):
    return 100 * count / denominator if denominator else None


def analyze(df):
    formatting, truncation, filtered = [], [], []
    for mode, label in (('none', 'raw'), ('strip', 'stripped'), ('host', 'host-only')):
        inputs = normalize_frame(df, mode)
        for y, cls in ((1, 'phishing'), (0, 'legitimate')):
            before = df[df.y == y]
            current = inputs[inputs.y == y]
            n = len(current)
            filtered.append(dict(input=label, class_name=cls, rows=len(before) - n))
            lengths = current.url.str.len()
            removed = lengths[lengths > 200] - 200
            truncation.append(dict(input=label, class_name=cls, denominator=n,
                exceeding_200=len(removed), percent=percent(len(removed), n),
                median_removed=float(removed.median()) if len(removed) else None))
            if mode == 'host':
                continue
            features = current.url.map(parsed_features)
            counts = dict(https_www=int(current.url.str.startswith('https://www.').sum()),
                          path=sum(f[0] for f in features), query=sum(f[1] for f in features))
            formatting.append(dict(input=label, class_name=cls, denominator=n,
                **{key: dict(count=value, percent=percent(value, n)) for key, value in counts.items()},
                parse_failures=sum(f[2] for f in features)))
    return dict(formatting=formatting, truncation=truncation, preprocessing_filtered=filtered)


def generate(data, manifest_path):
    df, cleaning = load_dataset(data)
    partitions, manifest = load_manifest(manifest_path, data, df)
    df['partition'] = partitions
    exploratory = df[df.partition.isin(('train', 'validation'))].copy()
    report = analyze(exploratory)
    # This deliberately separate check only reproduces a historical raw-length claim.
    affected = df[df.url.str.len() > 200]
    phishing = int(affected.y.sum())
    report.update(scope='pooled frozen training + validation only', max_len=200,
        dataset_sha256=manifest['dataset_sha256'], manifest_sha256=sha256(Path(manifest_path)),
        split_seed=manifest['seed'], holdout_type=manifest['holdout_type'], cleaning_counts=cleaning,
        selected_partitions={name: int((exploratory.partition == name).sum())
                             for name in ('train', 'validation')},
        historical_full_dataset_check=dict(scope='full deduplicated dataset; historical check only',
            denominator=len(df), raw_exceeding_200=len(affected), phishing=phishing,
            legitimate=len(affected) - phishing,
            matches_report=len(affected) == 1427 and phishing == 1427))
    if sha256(Path(data)) != manifest['dataset_sha256']:
        raise ValueError('dataset changed during analysis')
    return report


def display(report):
    def pct(value):
        return 'N/A' if value is None else f'{value:.3f}%'
    lines = ['# URL input diagnostics', '', 'Scope: pooled frozen training + validation only.', '',
        '| Input | Class | Denominator | https://www. count (%) | Nonempty path count (%) | Query count (%) |',
        '|---|---|---:|---:|---:|---:|']
    for row in report['formatting']:
        cells = [f"{row[k]['count']:,} ({pct(row[k]['percent'])})" for k in ('https_www', 'path', 'query')]
        lines.append(f"| {row['input']} | {row['class_name']} | {row['denominator']:,} | " + ' | '.join(cells) + ' |')
    lines += ['', '| Input | Class | Denominator | >200 characters count (%) | Median characters removed (affected only) |',
              '|---|---|---:|---:|---:|']
    for r in report['truncation']:
        median = 'N/A' if r['median_removed'] is None else f"{r['median_removed']:g}"
        lines.append(f"| {r['input']} | {r['class_name']} | {r['denominator']:,} | {r['exceeding_200']:,} ({pct(r['percent'])}) | {median} |")
    h = report['historical_full_dataset_check']
    lines += ['', f"Full-dataset historical check: {h['raw_exceeding_200']:,} of {h['denominator']:,} deduplicated raw URLs exceed 200 characters; {h['phishing']:,} phishing, {h['legitimate']:,} legitimate. Matches report: {h['matches_report']}.", '',
              'Rows filtered after preprocessing (input/class/count): ' + ', '.join(
                  f"{r['input']}/{r['class_name']}/{r['rows']}" for r in report['preprocessing_filtered']) + '.',
              'Formatting parse failures: ' + str(sum(r['parse_failures'] for r in report['formatting'])) + '.',
              'No model fitting, model selection, predictions or scores.']
    return '\n'.join(lines) + '\n'


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data', required=True)
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--out', type=Path, required=True, help='fresh aggregate-only output directory')
    a = ap.parse_args()
    try:
        report = generate(a.data, a.manifest)
        a.out.mkdir(parents=True, exist_ok=False)
        (a.out / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
        text = display(report)
        (a.out / 'REPORT.md').write_text(text)
    except (ValueError, OSError) as exc:
        ap.error(str(exc))
    print(text)


if __name__ == '__main__':
    main()
