"""Offline grouping, preserving the original hostname split algorithm."""
from functools import lru_cache
import hashlib
from importlib.metadata import version
import ipaddress
from pathlib import Path
import re
from urllib.parse import urlparse, urlsplit

from sklearn.model_selection import GroupShuffleSplit

PSL = Path(__file__).resolve().parent / 'suffixes/public_suffix_list.dat'


def host_of(url):
    u = url if "://" in url else "http://" + url
    try:
        h = (urlparse(u).hostname or "").lower()
    except ValueError:
        return ""
    return h[4:] if h.startswith("www.") else h


@lru_cache(maxsize=1)
def extractor():
    import tldextract
    return tldextract.TLDExtract(suffix_list_urls=(PSL.as_uri(),), cache_dir=None,
                                fallback_to_snapshot=False, include_psl_private_domains=True)


def parser_metadata():
    headers = [line[3:] for line in PSL.read_text().splitlines()
               if line.startswith(('// VERSION:', '// COMMIT:'))]
    return dict(parser='tldextract', parser_version=version('tldextract'),
                snapshot=headers, sha256=hashlib.sha256(PSL.read_bytes()).hexdigest(),
                private_suffixes=True, network=False)


@lru_cache(maxsize=300000)
def domain_of(url):
    """Return a namespaced grouping key and a status, never resolve a hostname."""
    try:
        parsed = urlsplit(url if '://' in url else 'http://' + url)
        host = (parsed.hostname or '').rstrip('.').lower()
        parsed.port  # validate malformed/non-numeric ports
        if not host:
            return 'malformed:', 'malformed'
        try:
            return 'ip:' + str(ipaddress.ip_address(host)), 'ip'
        except ValueError:
            pass
        host = host.encode('idna').decode('ascii')
        labels = host.split('.')
        if len(host) > 253 or any(not re.fullmatch(r'[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?', x)
                                  for x in labels):
            return 'malformed:', 'malformed'
        result = extractor()(host)
        if result.top_domain_under_public_suffix:
            return 'domain:' + result.top_domain_under_public_suffix, 'registrable'
        return 'host:' + host, 'no_registrable_suffix'
    except (ValueError, UnicodeError):
        return 'malformed:', 'malformed'


def split_frame(df, seed, group_col='host'):
    first = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=seed)
    trval_idx, test_idx = next(first.split(df, groups=df[group_col]))
    trval = df.iloc[trval_idx]
    second = GroupShuffleSplit(n_splits=1, test_size=0.15 / 0.85, random_state=seed)
    tr_i, va_i = next(second.split(trval, groups=trval[group_col]))
    parts = trval.iloc[tr_i], trval.iloc[va_i], df.iloc[test_idx]
    for left, right in ((0, 1), (0, 2), (1, 2)):
        assert not (set(parts[left][group_col]) & set(parts[right][group_col])), 'group leakage'
    return parts
