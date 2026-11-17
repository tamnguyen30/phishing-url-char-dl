"""Shared model-input transformations, applied before character truncation."""
import re

from splitting import host_of


def strip_prefix(url):
    """Remove scheme and one leading www.; preserve path and query spelling."""
    u = re.sub(r'^[a-zA-Z][a-zA-Z0-9+.-]*://', '', url)
    return re.sub(r'^www\.', '', u, flags=re.I)


def normalize_frame(df, mode):
    df = df.copy()
    if mode == 'host':
        df['url'] = df['host'] if 'host' in df else df.url.map(host_of)
    elif mode == 'strip':
        df['url'] = df.url.map(strip_prefix)
    elif mode != 'none':
        raise ValueError(f'unknown normalization: {mode}')
    if mode != 'none':
        df = df[df.url != ''].reset_index(drop=True)
    return df
