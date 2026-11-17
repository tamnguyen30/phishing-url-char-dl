"""Offline, aggregate-only leakage audit of hostname and registrable-domain splits."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import platform
from importlib.metadata import version
from pathlib import Path
import re
import sys
from urllib.parse import parse_qsl, urlsplit

from dataset import load_dataset
from provenance import command, gpu_details, sha256
from splitting import domain_of, host_of, parser_metadata, split_frame

ROOT = Path(__file__).resolve().parent
STATUSES = ('registrable', 'ip', 'no_registrable_suffix', 'malformed')
PAIRS = (("train", "validation"), ("train", "test"), ("validation", "test"))


def audit_keys(url):
    """No I/O. Exact key retains credentials/port/path/query/fragment literally."""
    if domain_of(url)[1] == 'malformed':
        return None, None, None
    parsed = urlsplit(url if '://' in url else 'http://' + url)
    host = parsed.hostname.lower()
    host = host[4:] if host.startswith('www.') else host
    if ':' in host:  # IPv6 brackets
        host = '[' + host + ']'
    userinfo = parsed.netloc.rsplit('@', 1)[0] + '@' if '@' in parsed.netloc else ''
    # Preserve explicit port spelling, including leading zeros.
    authority = parsed.netloc.rsplit('@', 1)[-1]
    tail = authority[authority.index(']') + 1:] if authority.startswith('[') else (
        ':' + authority.rsplit(':', 1)[1] if ':' in authority else '')
    # Preserve empty '?' and '#' delimiters, rather than reserializing parsed fields.
    without_scheme = re.sub(r'^[a-zA-Z][a-zA-Z0-9+.-]*://', '', url)
    suffix = without_scheme[len(parsed.netloc):]
    exact = userinfo + host + tail + suffix
    path = re.sub(r'[0-9]+', '{n}', parsed.path) if parsed.path else None
    query = tuple(sorted(key for key, value in parse_qsl(parsed.query, keep_blank_values=True))) if parsed.query else None
    return exact, path, query or None


def pair_matches(left, right, key):
    """Count shared distinct keys and participating rows; template keys need different domains."""
    def index(frame):
        result = defaultdict(Counter)
        for value, domain in zip(frame[key], frame.domain):
            if isinstance(value, (str, tuple)) and value:
                result[value][domain] += 1
        return result
    a, b = index(left), index(right)
    shared = 0
    left_urls = right_urls = 0
    for value in a.keys() & b.keys():
        if key == 'exact':
            shared += 1
            left_urls += sum(a[value].values())
            right_urls += sum(b[value].values())
        elif len(a[value]) > 1 or len(b[value]) > 1 or next(iter(a[value])) != next(iter(b[value])):
            shared += 1
            left_urls += sum(n for domain, n in a[value].items() if len(b[value]) > 1 or domain not in b[value])
            right_urls += sum(n for domain, n in b[value].items() if len(a[value]) > 1 or domain not in a[value])
    return dict(shared_keys=shared, left_urls=left_urls, right_urls=right_urls)


def audit(df, seed, method):
    parts = dict(zip(('train', 'validation', 'test'), split_frame(df, seed,
                'host' if method == 'hostname' else 'domain')))
    counts = {}
    for name, frame in parts.items():
        counts[name] = dict(urls=len(frame), phishing=int(frame.y.sum()),
            legitimate=int((frame.y == 0).sum()), hostnames=int(frame.host.nunique()),
            domain_group_keys=int(frame.domain.nunique()),
            registrable_domains=int(frame.loc[frame.domain_status == 'registrable', 'domain'].nunique()),
            grouping_status={status: int((frame.domain_status == status).sum()) for status in STATUSES},
            grouping_unique_keys={status: int(frame.loc[frame.domain_status == status, 'domain'].nunique())
                                  for status in STATUSES})
    pairs = {}
    for left, right in PAIRS:
        a, b = parts[left], parts[right]
        domains = set(a.loc[a.domain_status == 'registrable', 'domain']) & set(
            b.loc[b.domain_status == 'registrable', 'domain'])
        overlap = len(set(a.domain) & set(b.domain))
        if method == 'registrable':
            assert overlap == 0, 'registrable grouping leakage'
        pairs[f'{left}/{right}'] = dict(shared_registrable_domains=len(domains),
            shared_domain_group_keys=overlap,
            shared_hostnames=len(set(a.host) & set(b.host)),
            scheme_www_hostcase_exact=pair_matches(a, b, 'exact'),
            possible_path_similarities=pair_matches(a, b, 'path_template'),
            possible_query_similarities=pair_matches(a, b, 'query_template'))
    return dict(partitions=counts, pairs=pairs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--out', type=Path, default=Path('results_leakage'))
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44])
    args = parser.parse_args()
    if args.out.resolve() == (ROOT / 'results').resolve():
        parser.error('use a separate directory for new audit results')
    if args.out.exists():
        parser.error('output directory must be new to preserve prior results')
    starting_git = dict(commit=command(['git', 'rev-parse', 'HEAD'], ROOT).strip(),
        status=command(['git', 'status', '--porcelain=v1', '--untracked-files=all'], ROOT))
    checksum = sha256(args.data)
    snapshot = command([sys.executable, '-m', 'pip', 'freeze', '--all'])
    df, cleaning = load_dataset(args.data)
    if sha256(args.data) != checksum:
        raise RuntimeError('Input changed while reading')
    df['host'] = df.url.map(host_of)
    domains = df.url.map(domain_of)
    df['domain'] = domains.map(lambda x: x[0])
    df['domain_status'] = domains.map(lambda x: x[1])
    keys = df.url.map(audit_keys)
    for i, name in enumerate(('exact', 'path_template', 'query_template')):
        df[name] = keys.map(lambda x: x[i])
    report = dict(recorded_at=datetime.now(timezone.utc).isoformat(),
        input_sha256=checksum, git=starting_git, parser=parser_metadata(),
        python=sys.version, os=platform.platform(),
        dependency_versions={name: version(name) for name in ("pandas", "numpy", "scikit-learn", "tldextract")},
        gpu=gpu_details(), cleaning=cleaning,
        grouping_status={status: int((df.domain_status == status).sum()) for status in STATUSES},
        grouping_unique_keys={status: int(df.loc[df.domain_status == status, 'domain'].nunique())
                              for status in STATUSES},
        exact_audit_excluded_malformed_urls=int((df.domain_status == 'malformed').sum()),
        normalization='none; original deduplicated URLs before model normalization',
        audits={method: {str(seed): audit(df, seed, method) for seed in args.seeds}
                for method in ('hostname', 'registrable')})
    args.out.mkdir(parents=True)
    (args.out / 'audit.json').write_text(json.dumps(report, indent=2) + '\n')
    (args.out / 'requirements-environment.txt').write_text(snapshot)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
