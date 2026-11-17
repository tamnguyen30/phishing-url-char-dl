"""Frozen assignment integrity and preprocessing invariance, without model scores."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from dataset import load_dataset
from frozen_splits import freeze, load_manifest, NAMES, summary
from train import load_and_split


class FrozenTests(unittest.TestCase):
    def test_roundtrip_inputs_and_rejections(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = Path(tmp) / 'fixture.csv'
            data.write_text('URL,label\n' + ''.join(
                f'https://sub{j}.site{i}.com/path,{i % 2}\n'
                for i in range(40) for j in range(2)) +
                'http://[broken,0\nhttps://,1\n https://sub0.site0.com/path ,0\n,0\n')
            path = Path(tmp) / 'fixture.split-manifest.json'
            m = freeze(data, path)
            df, _ = load_dataset(data)
            assigned, _ = load_manifest(path, data, df)
            expected = dict(zip(df.url, assigned))
            self.assertEqual(len(assigned), len(df))
            self.assertEqual(summary(m['assignments'])[1],
                             {'train/validation': 0, 'train/test': 0, 'validation/test': 0})
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)
            with self.assertRaises(FileExistsError):
                freeze(data, path)
            reference = None
            for model in ('tfidf_lr', 'charcnn', 'bilstm'):
                for norm in ('none', 'strip', 'host'):
                    a = SimpleNamespace(data=data, url_col='URL', label_col='label', phish_label=0,
                        max_rows=0, seed=99, normalize=norm, group_by='registrable',
                        split_manifest=path, model=model)
                    with contextlib.redirect_stdout(io.StringIO()):
                        parts = load_and_split(a)
                    actual = {h: name for name, part in zip(NAMES, parts) for h in part.url_sha256}
                    if norm == 'none':
                        reference = actual
                        self.assertEqual([len(p) for p in parts], [list(expected.values()).count(n) for n in NAMES])
                    self.assertTrue(all(reference[h] == n for h, n in actual.items()))
                    self.assertEqual(sum(a.preprocessing_filtered.values()), len(df) - sum(map(len, parts)))
                    for x, y in ((0, 1), (0, 2), (1, 2)):
                        self.assertFalse(set(parts[x].domain) & set(parts[y].domain))
            # All integrity checks use real loader data and manipulated on-disk artifacts.
            for mutate in (
                lambda x: x.update(dataset_sha256='wrong'),
                lambda x: x['assignments'].pop(),
                lambda x: x['assignments'][0].update(group='wrong'),
                lambda x: x['assignments'][0].update(partition='unknown'),
                lambda x: x['counts']['train'].update(rows=-1),
                lambda x: x['policy'].update(group_by='hostname'),
            ):
                damaged = json.loads(json.dumps(m))
                mutate(damaged)
                path.write_text(json.dumps(damaged))
                with self.assertRaisesRegex(ValueError, 'split manifest mismatch'):
                    load_manifest(path, data, df)
            path.write_text(json.dumps(m))
            data.write_text(data.read_text() + 'https://new.com,1\n')
            with self.assertRaisesRegex(ValueError, 'checksum'):
                load_manifest(path, data, df)


if __name__ == '__main__':
    unittest.main()
