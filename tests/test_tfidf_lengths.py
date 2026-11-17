"""Exercise both real baseline fits and validate paired input/split isolation."""
from pathlib import Path
import json
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from dataset import load_dataset
from frozen_splits import freeze, load_manifest
from compare_tfidf_lengths import prepare, compare
from tfidf_baseline import estimators


class LengthComparisonTests(unittest.TestCase):
    def test_independent_fits_same_rows_and_training_only_vocabulary(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = Path(tmp) / 'fixture.csv'
            data.write_text('URL,label\n' + ''.join(
                f'https://www.site{i}.com/' + 'x' * 220 +
                ('/phishing-tail' if i % 2 == 0 else '/legitimate-tail') + f',{i % 2}\n'
                for i in range(80)))
            manifest = Path(tmp) / 'fixture.split-manifest.json'
            freeze(data, manifest)
            df, _ = load_dataset(data)
            assigned, _ = load_manifest(manifest, data, df)
            with patch('compare_tfidf_lengths.normalize_frame', wraps=__import__('preprocessing').normalize_frame) as norm:
                parts, m, filtered = prepare(data, manifest)
            self.assertEqual(set(norm.call_args.args[0].partition), {'train', 'validation'})
            self.assertEqual(sum(len(p) for p in parts.values()), sum(n != 'test' for n in assigned))
            self.assertEqual(filtered, {'train': 0, 'validation': 0})
            self.assertTrue(all(not u.startswith('https://') for u in parts['train'].url))
            snapshots = {n: p.copy(deep=True) for n, p in parts.items()}
            fitted = []
            def factory():
                vec, clf = estimators()
                original_fit = vec.fit_transform
                original_transform = vec.transform
                seen = dict(vec=vec, clf=clf)
                def fit(inputs):
                    seen['train_inputs'] = list(inputs)
                    return original_fit(inputs)
                def transform(inputs):
                    seen['val_inputs'] = list(inputs)
                    return original_transform(inputs)
                vec.fit_transform = fit
                vec.transform = transform
                fitted.append(seen)
                return vec, clf
            with patch('compare_tfidf_lengths.estimators', side_effect=factory):
                results = compare(parts)
            self.assertEqual(len(fitted), 2)
            self.assertIsNot(fitted[0]['vec'], fitted[1]['vec'])
            for key, part in (('train_inputs', 'train'), ('val_inputs', 'validation')):
                self.assertEqual(fitted[0][key], parts[part].url.tolist())
                self.assertEqual(fitted[1][key], [u[:200] for u in parts[part].url])
            self.assertIn('tail', fitted[0]['vec'].vocabulary_)
            self.assertNotIn('tail', fitted[1]['vec'].vocabulary_)
            for n, p in parts.items():
                __import__('pandas').testing.assert_frame_equal(p, snapshots[n])
            self.assertEqual(fitted[0]['clf'].get_params(), fitted[1]['clf'].get_params())
            for metric in ('precision', 'recall', 'f1'):
                for run in ('full', 'first_200'):
                    self.assertTrue(0 <= results[run]['validation_phishing'][metric] <= 1)
                self.assertAlmostEqual(results['change_200_minus_full'][metric],
                    results['first_200']['validation_phishing'][metric] - results['full']['validation_phishing'][metric])
            self.assertFalse(results['full']['convergence_warnings'])
            self.assertFalse(results['first_200']['convergence_warnings'])
            out = Path(tmp) / 'comparison'
            subprocess.run([sys.executable, str(Path(__file__).resolve().parents[1] / 'compare_tfidf_lengths.py'),
                '--data', str(data), '--manifest', str(manifest), '--out', str(out)],
                check=True, capture_output=True, text=True)
            saved = json.loads((out / 'comparison.json').read_text())
            self.assertEqual(saved['results'], results)
            self.assertEqual(saved['vectorizer_params']['dtype'], 'float64')
            self.assertTrue((out / 'REPORT.md').exists())


if __name__ == '__main__':
    unittest.main()
