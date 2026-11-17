"""Private fixed domain assignments. CLI freezes/verifies only; never trains or scores."""
import argparse
import hashlib
import json
import os
from pathlib import Path

from dataset import load_dataset
from provenance import sha256
from splitting import domain_of, parser_metadata, split_frame

NAMES = ('train', 'validation', 'test')
HOLDOUT = 'same-dataset robustness holdout'


def url_hash(url):
    return hashlib.sha256(url.encode('utf-8')).hexdigest()


def policy():
    return dict(group_by='registrable', parser=parser_metadata(),
                splitter='GroupShuffleSplit 0.15 then 0.15/0.85',
                cleaning='drop missing URL/label; strip whitespace; first exact URL duplicate',
                url_hash='sha256 UTF-8 cleaned original URL')


def records(df):
    return [dict(url_sha256=url_hash(u), group=domain_of(u)[0]) for u in df.url]


def summary(rows):
    groups = {name: {r['group'] for r in rows if r['partition'] == name} for name in NAMES}
    overlap = {f'{a}/{b}': len(groups[a] & groups[b])
               for a, b in (('train', 'validation'), ('train', 'test'), ('validation', 'test'))}
    if any(overlap.values()):
        raise ValueError('split manifest group overlap is nonzero')
    return {n: dict(rows=sum(r['partition'] == n for r in rows), groups=len(groups[n]))
            for n in NAMES}, overlap


def load_manifest(path, data, df, url_col='URL', label_col='label', phish_label=0):
    try:
        manifest = json.loads(Path(path).read_text())
        if manifest['version'] != 1 or manifest['dataset_sha256'] != sha256(Path(data)):
            raise ValueError('dataset checksum or manifest version disagrees')
        if manifest['policy'] != policy():
            raise ValueError('grouping/parser/cleaning policy disagrees')
        if manifest['columns'] != [url_col, label_col, phish_label]:
            raise ValueError('dataset columns or class mapping disagrees')
        rows = manifest['assignments']
        expected = records(df)
        indexed = {r['url_sha256']: r for r in rows}
        if len(indexed) != len(rows) or len(rows) != len(expected):
            raise ValueError('duplicate or missing URL assignments')
        ordered = []
        for r in expected:
            saved = indexed.get(r['url_sha256'])
            if saved is None or saved['group'] != r['group'] or saved['partition'] not in NAMES:
                raise ValueError('URL/group/partition disagrees with dataset')
            ordered.append(saved)
        counts, overlap = summary(rows)
        if counts != manifest['counts'] or overlap != manifest['pairwise_group_overlap']:
            raise ValueError('split counts disagree')
        if manifest['holdout_type'] != HOLDOUT:
            raise ValueError('unsupported holdout type')
        return [r['partition'] for r in ordered], manifest
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f'split manifest mismatch: {exc}') from exc


def freeze(data, path, seed=42, url_col='URL', label_col='label', phish_label=0):
    checksum = sha256(Path(data))
    df, cleaning_counts = load_dataset(data, url_col, label_col, phish_label)
    df['domain'] = df.url.map(lambda u: domain_of(u)[0])
    parts = split_frame(df, seed, 'domain')
    rows = []
    for name, part in zip(NAMES, parts):
        rows.extend(dict(**r, partition=name) for r in records(part))
    counts, overlap = summary(rows)
    if sha256(Path(data)) != checksum:
        raise ValueError('dataset changed during freezing')
    manifest = dict(version=1, dataset_sha256=checksum, seed=seed, policy=policy(),
                    columns=[url_col, label_col, phish_label], holdout_type=HOLDOUT,
                    prior_use='README documents full-CSV experiments across seeds 42,43,44; untouched domains unverified',
                    cleaning_counts=cleaning_counts, counts=counts,
                    pairwise_group_overlap=overlap, assignments=rows)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation protects existing freezes; mode 0600 because group keys are private.
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, 'w') as stream:
        json.dump(manifest, stream)
        stream.write('\n')
    assignments, loaded = load_manifest(path, data, df, url_col, label_col, phish_label)
    expected = {r['url_sha256']: r['partition'] for r in rows}
    if assignments != [expected[url_hash(u)] for u in df.url]:
        raise ValueError('round-trip assignment mismatch')
    return loaded


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data', required=True)
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--verify', action='store_true')
    a = ap.parse_args()
    if a.verify:
        df, _ = load_dataset(a.data)
        _, m = load_manifest(a.manifest, a.data, df)
    else:
        m = freeze(a.data, a.manifest, a.seed)
    print(json.dumps({k: m[k] for k in ('dataset_sha256', 'seed', 'holdout_type', 'counts', 'pairwise_group_overlap')}, indent=2))
    print('Every deduplicated URL assignment verified; no model training or scoring.')


if __name__ == '__main__':
    main()
