"""Integration coverage for aggregate-only provenance and the shared loader."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace

from dataset import load_dataset

ROOT = Path(__file__).resolve().parents[1]


class ProvenanceTest(unittest.TestCase):
    def test_fixture(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            data = base / "fixture.csv"
            data.write_text('URL,label\n https://bad.invalid/a ,0\nhttps://bad.invalid/a,1\n'
                            'https://good.invalid,1\n,0\nhttps://missing.invalid,\n00123,0\n')
            frame, counts = load_dataset(data)
            self.assertEqual(counts["original"], dict(rows=6, phishing=3,
                             legitimate=2, missing_labels=1))
            self.assertEqual(counts["non_missing"]["rows"], 4)
            self.assertEqual(counts["deduplicated"]["rows"], 3)
            self.assertEqual(frame.url.tolist()[-1], "00123")
            self.assertEqual(frame.y.tolist(), [1, 0, 1])
            out = base / "capture"
            subprocess.run([sys.executable, str(ROOT / "provenance.py"),
                            "--data", str(data), "--out", str(out)], check=True)
            record = json.loads((out / "provenance.json").read_text())
            self.assertEqual(record["counts"], counts)
            self.assertEqual(record["input"]["sha256"], hashlib.sha256(data.read_bytes()).hexdigest())
            self.assertEqual(record["difference_from_reference"]["original_rows"], 6 - 235795)
            self.assertTrue(record["git"]["commit"])
            self.assertIn("status", record["git"])
            self.assertIn("pandas==", (out / "requirements-environment.txt").read_text().lower())
            self.assertEqual(sorted(p.name for p in out.iterdir()),
                             ["provenance.json", "requirements-environment.txt"])
            self.assertNotIn("bad.invalid", (out / "provenance.json").read_text())

    def test_training_uses_shared_loader(self):
        from train import load_and_split
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory) / "fixture.csv"
            data.write_text("URL,label\n" + "".join(
                f"https://host{i}.invalid/a,{i % 2}\n" for i in range(40)) +
                " https://host0.invalid/a ,0\n")
            splits = load_and_split(SimpleNamespace(data=data, url_col="URL", label_col="label",
                phish_label=0, max_rows=0, seed=42, normalize="none"))
            self.assertEqual(sum(map(len, splits)), 40)
            for split in splits:
                self.assertTrue((split.y == (split.label == 0).astype(int)).all())


if __name__ == "__main__":
    unittest.main()
