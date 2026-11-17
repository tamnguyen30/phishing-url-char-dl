"""Known formatting/length counts and frozen-scope integration; no model fitting."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from dataset import load_dataset
from frozen_splits import freeze, load_manifest
from preprocessing import normalize_frame
from summarize_inputs import analyze, generate, parsed_features, parse_input


class InputEffectsTests(unittest.TestCase):
    def test_exact_counts_and_transform_before_truncation(self):
        prefix = 'https://www.example.com/'
        long = prefix + 'a' * (210 - len(prefix))
        boundary = 'https://other.com/' + 'b' * (200 - len('https://other.com/'))
        df = pd.DataFrame({'url': [long, boundary, 'https://www.good.com',
                                  'http://good.net/?q=1', 'http://[broken'],
                           'y': [1, 1, 0, 0, 1]})
        report = analyze(df)
        f = {(r['input'], r['class_name']): r for r in report['formatting']}
        raw = f['raw', 'phishing']
        self.assertEqual(raw['denominator'], 3)
        self.assertEqual(raw['https_www']['count'], 1)
        self.assertEqual(raw['path']['count'], 2)
        self.assertAlmostEqual(raw['path']['percent'], 200 / 3)
        self.assertEqual(raw['parse_failures'], 1)
        stripped = f['stripped', 'phishing']
        self.assertEqual(stripped['path']['count'], 2)
        self.assertEqual(stripped['https_www']['count'], 0)
        self.assertEqual(f['raw', 'legitimate']['query']['count'], 1)
        self.assertEqual(f['stripped', 'legitimate']['query']['count'], 1)
        parsed = parse_input('example.com:8080/a?x=1')
        self.assertEqual((parsed.hostname, parsed.path, parsed.query), ('example.com', '/a', 'x=1'))
        self.assertEqual(parsed_features('example.com:8080/a?x=1'), (True, True, False))
        self.assertEqual(parsed_features('example.com'), (False, False, False))
        self.assertEqual(parsed_features('example.com?'), (False, False, False))
        t = {(r['input'], r['class_name']): r for r in report['truncation']}
        self.assertEqual(t['raw', 'phishing']['exceeding_200'], 1)
        self.assertEqual(t['raw', 'phishing']['median_removed'], 10)
        # Removing https://www. BEFORE truncation makes this 198 characters.
        self.assertEqual(t['stripped', 'phishing']['exceeding_200'], 0)
        self.assertIsNone(t['stripped', 'phishing']['median_removed'])
        self.assertEqual(t['host-only', 'phishing']['denominator'], 2)
        self.assertEqual(t['host-only', 'phishing']['exceeding_200'], 0)
        self.assertEqual(sum(r['rows'] for r in report['preprocessing_filtered']), 1)
        self.assertEqual(len(normalize_frame(df, 'strip').url.iloc[0]), 198)
        # Median counts affected URLs only, including fractional medians.
        median = analyze(pd.DataFrame({'url': ['x' * 201, 'x' * 204, 'short'], 'y': [1, 1, 1]}))
        self.assertEqual(median['truncation'][0]['median_removed'], 2.5)

    def test_frozen_scope_and_full_dataset_check_separated(self):
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory) / 'fixture.csv'
            data.write_text('URL,label\n' + ''.join(
                f'https://site{i}.com/' + 'a' * 210 + f',{i % 2}\n' for i in range(40)))
            manifest = Path(directory) / 'fixture.split-manifest.json'
            freeze(data, manifest)
            df, _ = load_dataset(data)
            partitions, _ = load_manifest(manifest, data, df)
            # Observe the actual frame passed to exploratory analysis.
            with patch('summarize_inputs.analyze', wraps=analyze) as spy:
                report = generate(data, manifest)
            selected = spy.call_args.args[0]
            self.assertEqual(set(selected.partition), {'train', 'validation'})
            self.assertEqual(len(selected), sum(p != 'test' for p in partitions))
            self.assertEqual(sum(r['denominator'] for r in report['truncation'][:2]), len(selected))
            self.assertEqual(report['historical_full_dataset_check']['raw_exceeding_200'], 40)
            self.assertFalse(report['historical_full_dataset_check']['matches_report'])
            self.assertNotIn('site0.com', json.dumps(report))
            data.write_text(data.read_text() + 'https://new.com,0\n')
            with self.assertRaisesRegex(ValueError, 'checksum'):
                generate(data, manifest)


if __name__ == '__main__':
    unittest.main()
