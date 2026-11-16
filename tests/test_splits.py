"""Grouping semantics, legacy compatibility, and offline audit checks."""
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import pandas as pd

from audit_splits import audit, audit_keys, pair_matches
from splitting import domain_of, extractor, host_of, parser_metadata
from train import load_and_split

ROOT = Path(__file__).resolve().parents[1]


class GroupTests(unittest.TestCase):
    def test_offline_psl_and_fallbacks(self):
        domain_of.cache_clear()
        extractor.cache_clear()
        with patch.object(socket.socket, 'connect', side_effect=AssertionError('network attempted')):
            self.assertEqual(domain_of('https://shop.example.co.uk')[0], 'domain:example.co.uk')
            self.assertEqual(domain_of('https://login.example.co.uk')[0], 'domain:example.co.uk')
            self.assertNotEqual(domain_of('https://a.blogspot.com')[0], domain_of('https://b.blogspot.com')[0])
            self.assertEqual(domain_of('http://www.BÜCHER.de.')[0], 'domain:xn--bcher-kva.de')
            self.assertEqual(domain_of('http://[2001:0db8::1]')[0], 'ip:2001:db8::1')
            self.assertEqual(domain_of('http://127.0.0.1')[1], 'ip')
            self.assertEqual(domain_of('http://localhost'), ('host:localhost', 'no_registrable_suffix'))
            self.assertEqual(domain_of('https://co.uk')[1], 'no_registrable_suffix')
            for url in ('http://[broken', 'http://bad_host.com', '/relative', 'http://x.com:bad'):
                self.assertEqual(domain_of(url), ('malformed:', 'malformed'))
            # PSL wildcard and exception rules (not an approximate last-two-label parser).
            self.assertEqual(domain_of('http://a.b.ck')[0], 'domain:a.b.ck')
            self.assertEqual(domain_of('http://a.www.ck')[0], 'domain:www.ck')
        self.assertEqual(parser_metadata()['sha256'], 'b69315c085d53972724b8f2df111ffc329b0c84fe0a47d62c8c91655cc774a38')

    def test_legacy_training_unchanged(self):
        # Load the exact previous committed training code for an independent comparison.
        source = subprocess.check_output(['git', 'show', 'b99c48d:train.py'], cwd=ROOT, text=True)
        legacy = {'__name__': 'legacy_train'}
        exec(compile(source, 'legacy_train.py', 'exec'), legacy)
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory) / 'fixture.csv'
            data.write_text('URL,label\n' + ''.join(
                f'https://sub{i % 2}.host{i // 2}.com/a,{i % 2}\n' for i in range(80)) +
                'http://[broken,0\n')
            for norm in ('none', 'strip', 'host'):
                for seed in (42, 43, 44):
                    args = SimpleNamespace(data=data, url_col='URL', label_col='label',
                        phish_label=0, max_rows=0, seed=seed, normalize=norm, group_by='hostname')
                    expected = legacy['load_and_split'](args)
                    actual = load_and_split(args)
                    for a, b in zip(actual, expected):
                        pd.testing.assert_frame_equal(a, b)
                    args.group_by = 'registrable'
                    grouped = load_and_split(args)
                    for a, b in ((0, 1), (0, 2), (1, 2)):
                        self.assertFalse(set(grouped[a].domain) & set(grouped[b].domain))
                    self.assertEqual(sum(map(len, grouped)), sum(map(len, actual)))

    def test_exact_and_templates(self):
        self.assertEqual(audit_keys('https://WWW.Example.com/AbC?a=1#x')[0],
                         audit_keys('http://example.COM/AbC?a=1#x')[0])
        self.assertNotEqual(audit_keys('https://example.com/AbC')[0],
                            audit_keys('https://example.com/abc')[0])
        self.assertIsNone(audit_keys('http://bad_host.com/path')[0])
        self.assertEqual(audit_keys('https://a.com/item/123?q=secret&b=2')[1:],
                         ('/item/{n}', ('b', 'q')))
        self.assertEqual(audit_keys('https://a.com/')[1:], ('/', None))
        self.assertEqual(audit_keys('https://a.com')[1:], (None, None))
        # Removing www can cross a PSL suffix-only/registrable boundary.
        self.assertEqual(audit_keys('http://co.uk/login')[0],
                         audit_keys('https://www.co.uk/login')[0])
        self.assertNotEqual(domain_of('http://co.uk/login')[0],
                            domain_of('https://www.co.uk/login')[0])
        self.assertNotEqual(audit_keys('https://a.com?')[0], audit_keys('https://a.com')[0])
        a = pd.DataFrame({'domain': ['domain:a.com', 'domain:a.com'],
                          'path_template': ['/login', '/login'], 'exact': ['key', 'key']})
        b = pd.DataFrame({'domain': ['domain:b.com', 'domain:a.com'],
                          'path_template': ['/login', '/login'], 'exact': ['key', 'other']})
        self.assertEqual(pair_matches(a, b, 'path_template'),
                         dict(shared_keys=1, left_urls=2, right_urls=1))
        self.assertEqual(pair_matches(a, b, 'exact'), dict(shared_keys=1, left_urls=2, right_urls=1))
        self.assertEqual(pair_matches(a, a, 'path_template')['shared_keys'], 0)
        # pandas string columns may turn None into NaN: missing templates never match.
        empty = pd.DataFrame({'domain': ['domain:a.com', 'domain:b.com'],
                              'path_template': pd.Series([None, None], dtype='str')})
        self.assertEqual(pair_matches(empty.iloc[:1], empty.iloc[1:], 'path_template'),
                         dict(shared_keys=0, left_urls=0, right_urls=0))

    def test_audit_detects_subdomain_overlap(self):
        urls = [f'https://sub{i}.example.com/login/{i}?token={i}' for i in range(30)]
        urls += [f'https://site{i}.org/login/{i}?token={i}' for i in range(30)]
        df = pd.DataFrame({'url': urls, 'y': [0, 1] * 30})
        df['host'] = df.url.map(host_of)
        df['domain'] = df.url.map(lambda x: domain_of(x)[0])
        df['domain_status'] = df.url.map(lambda x: domain_of(x)[1])
        keys = df.url.map(audit_keys)
        for i, column in enumerate(('exact', 'path_template', 'query_template')):
            df[column] = keys.map(lambda x: x[i])
        original = audit(df, 42, 'hostname')
        self.assertTrue(any(pair['shared_registrable_domains'] for pair in original['pairs'].values()))
        grouped = audit(df, 42, 'registrable')
        self.assertTrue(all(pair['shared_domain_group_keys'] == 0 for pair in grouped['pairs'].values()))
        self.assertTrue(any(pair['possible_path_similarities']['shared_keys'] for pair in grouped['pairs'].values()))
        self.assertEqual(sum(p['urls'] for p in grouped['partitions'].values()), 60)


if __name__ == '__main__':
    unittest.main()
