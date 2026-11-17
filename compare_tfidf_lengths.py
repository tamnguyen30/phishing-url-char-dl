"""Paired stripped full/200-character TF-IDF comparison, validation only."""
import argparse
from importlib.metadata import version
import json
from pathlib import Path
import platform
import random
import sys
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import precision_score, recall_score, f1_score
from threadpoolctl import threadpool_limits

from dataset import load_dataset
from frozen_splits import load_manifest, url_hash
from preprocessing import normalize_frame
from provenance import command, sha256
from tfidf_baseline import estimators


def prepare(data, manifest_path):
    df, _ = load_dataset(data)
    assignments, manifest = load_manifest(manifest_path, data, df)
    df['partition'] = assignments
    # Discard held-out rows before normalization, fitting, or prediction.
    df = df[df.partition.isin(('train', 'validation'))].copy()
    df['url_sha256'] = df.url.map(url_hash)
    before = df.partition.value_counts().to_dict()
    df = normalize_frame(df, 'strip')
    parts = {name: df[df.partition == name].copy() for name in ('train', 'validation')}
    filtered = {name: before.get(name, 0) - len(part) for name, part in parts.items()}
    for name, part in parts.items():
        if set(part.y) != {0, 1}:
            raise ValueError(f'{name} must contain both classes')
    return parts, manifest, filtered


def compare(parts, seed=42):
    results = {}
    for name, length in (('full', None), ('first_200', 200)):
        random.seed(seed)
        np.random.seed(seed)
        train, val = parts['train'], parts['validation']
        tr_inputs = train.url if length is None else train.url.str.slice(0, length)
        va_inputs = val.url if length is None else val.url.str.slice(0, length)
        vec, clf = estimators()
        # Identical limits for reproducibility and to avoid BLAS oversubscription.
        with threadpool_limits(limits=1), warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always', ConvergenceWarning)
            Xtr = vec.fit_transform(tr_inputs)  # independent vocabulary AND IDF per run
            clf.fit(Xtr, train.y)
            probabilities = clf.predict_proba(vec.transform(va_inputs))[:, list(clf.classes_).index(1)]
        predicted = probabilities >= 0.5
        results[name] = dict(validation_phishing=dict(
            precision=float(precision_score(val.y, predicted, zero_division=0)),
            recall=float(recall_score(val.y, predicted, zero_division=0)),
            f1=float(f1_score(val.y, predicted, zero_division=0))),
            vocabulary_size=len(vec.vocabulary_), iterations=clf.n_iter_.tolist(),
            convergence_warnings=[str(w.message) for w in caught if issubclass(w.category, ConvergenceWarning)],
            affected_rows={n: int((p.url.str.len() > 200).sum()) if length else 0 for n, p in parts.items()})
    results['change_200_minus_full'] = {k: results['first_200']['validation_phishing'][k] -
        results['full']['validation_phishing'][k] for k in ('precision', 'recall', 'f1')}
    return results


def render(report):
    lines = ['# Stripped TF-IDF length comparison', '',
        'Real CSV comparison; frozen training and validation only. Test holdout was not fitted, transformed, predicted or scored.', '',
        '| Stripped input | Validation phishing precision | Recall | F1 |', '|---|---:|---:|---:|']
    for name, label in (('full', 'Full length'), ('first_200', 'First 200 characters')):
        r = report['results'][name]['validation_phishing']
        lines.append(f"| {label} | {r['precision']:.6f} | {r['recall']:.6f} | {r['f1']:.6f} |")
    d = report['results']['change_200_minus_full']
    lines.append(f"| Change (200 − full) | {d['precision']:+.6f} | {d['recall']:+.6f} | {d['f1']:+.6f} |")
    lines += ['', f"Split seed: {report['split_seed']}; model RNG seed: {report['model_seed']}; threshold: 0.5.",
        f"Manifest SHA-256: `{report['manifest_sha256']}`.",
        f"Dataset SHA-256: `{report['dataset_sha256']}`.", '',
        '| Partition | Rows | Phishing | Legitimate | Filtered after stripping |', '|---|---:|---:|---:|---:|']
    for n, c in report['counts'].items():
        lines.append(f"| {n} | {c['rows']:,} | {c['phishing']:,} | {c['legitimate']:,} | {report['filtered'][n]} |")
    lines += ['', 'Both runs use independent training-only vocabularies and IDF, the existing baseline hyperparameters, and the same rows and labels. No hyperparameter selection.', '',
        'Exact command used (from repository root):', '', '```bash', report['reproduction_command'], '```', '',
        'See comparison.json for full estimator settings, versions, convergence diagnostics, row-assignment digests and unrounded metrics.']
    return '\n'.join(lines) + '\n'


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data', required=True)
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--out', type=Path, required=True, help='fresh output directory')
    ap.add_argument('--seed', type=int, default=42, help='model RNG seed; split comes from manifest')
    a = ap.parse_args()
    if a.out.exists():
        ap.error('output directory already exists; use a fresh directory')
    try:
        parts, manifest, filtered = prepare(a.data, a.manifest)
        vec, clf = estimators()
        report = dict(scope='frozen train/validation only; test sealed',
            dataset_sha256=manifest['dataset_sha256'], manifest_sha256=sha256(Path(a.manifest)),
            split_seed=manifest['seed'], model_seed=a.seed, holdout_type=manifest['holdout_type'],
            threshold=0.5, positive_class='phishing (original label=0)',
            counts={n: dict(rows=len(p), phishing=int(p.y.sum()), legitimate=int((p.y == 0).sum())) for n, p in parts.items()},
            assignment_label_digests={n: url_hash('\n'.join(f'{h}:{y}' for h, y in zip(p.url_sha256, p.y))) for n, p in parts.items()},
            filtered=filtered, vectorizer_params=dict(vec.get_params(), dtype=np.dtype(vec.dtype).name), classifier_params=clf.get_params(),
            threads=1, python=platform.python_version(),
            dependencies={n: version(n) for n in ('numpy', 'pandas', 'scipy', 'scikit-learn', 'threadpoolctl', 'tldextract')},
            git_commit=command(['git', 'rev-parse', 'HEAD']).strip(),
            reproduction_command=__import__('shlex').join([sys.executable, *sys.argv]))
        report['results'] = compare(parts, a.seed)
        if sha256(Path(a.data)) != manifest['dataset_sha256']:
            raise ValueError('dataset changed during comparison')
        a.out.mkdir(parents=True, exist_ok=False)
        (a.out / 'comparison.json').write_text(json.dumps(report, indent=2) + '\n')
        text = render(report)
        (a.out / 'REPORT.md').write_text(text)
        print(text)
    except (ValueError, OSError) as exc:
        ap.error(str(exc))


if __name__ == '__main__':
    main()
