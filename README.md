# Deep Learning for Phishing URL Detection Using Character-Level Features

COSC 50523 course project (preliminary report code). We compare a TF-IDF + logistic regression
baseline with a character-level CNN and a character-level BiLSTM on raw URL strings.

## Setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Data (not included)
Download the PhiUSIIL Phishing URL dataset (UCI ML Repository) and place the CSV at
`data/PhiUSIIL_Phishing_URL_Dataset.csv`. Check the license on the dataset page.
Only the `URL` and `label` columns are used (`label=0` means phishing, `label=1` legitimate).

## Run
```bash
# single experiment
python train.py --data data/PhiUSIIL_Phishing_URL_Dataset.csv --model charcnn --normalize strip

# all models x normalizations x seeds (27 runs; skips finished runs)
python run_all.py --data data/PhiUSIIL_Phishing_URL_Dataset.csv

# aggregate results (mean +/- std over seeds) -> results/summary.csv
python summarize.py

# quick smoke test
python train.py --data data/PhiUSIIL_Phishing_URL_Dataset.csv --model charcnn --normalize strip --max_rows 5000 --epochs 2 --out results_debug
```
Models: `tfidf_lr`, `charcnn`, `bilstm`. Per-run results are written to `results/*.json`
(test metrics, validation curve, timing, full config).

## Normalization options
- `none`: raw URL
- `strip`: remove the scheme (`https://`) and a leading `www.`, keep the path
- `host`: hostname only (also removes the path)

In this dataset every benign URL is a bare `https://www.<domain>` homepage, so raw-URL scores are
inflated by formatting shortcuts. `strip` and `host` remove those cues.

## Protocol
- Exact duplicate URLs removed.
- 70/15/15 train/val/test split grouped by hostname (no host appears in two splits; asserted in code).
- Model selection on validation AUROC; test set touched once per run.
- Seeds 42, 43, 44. Each seed changes both the split and the model initialization.

## Results (test set, mean ± std over 3 seeds)
| norm | model | F1 | AUROC |
|---|---|---|---|
| none | tfidf_lr | 0.9968 ± 0.0002 | 0.9989 ± 0.0001 |
| none | charcnn | 0.9978 ± 0.0003 | 0.9989 ± 0.0002 |
| none | bilstm | 0.9979 ± 0.0001 | 0.9990 ± 0.0002 |
| strip | tfidf_lr | 0.8833 ± 0.0016 | 0.9469 ± 0.0014 |
| strip | charcnn | 0.8843 ± 0.0024 | 0.9462 ± 0.0011 |
| strip | bilstm | 0.8847 ± 0.0021 | 0.9465 ± 0.0006 |
| host | tfidf_lr | 0.8367 ± 0.0035 | 0.9220 ± 0.0028 |
| host | charcnn | 0.8314 ± 0.0062 | 0.9206 ± 0.0007 |
| host | bilstm | 0.8320 ± 0.0066 | 0.9215 ± 0.0019 |

Full table (precision, recall, etc.) is in `results/summary.csv`.

## Not committed
Datasets, checkpoints, weights, and per-URL predictions (see `.gitignore`).

## AI assistance
Starter code was assisted by Claude (Anthropic). Experiments were run and analyzed by the team.
## Reproduce dataset and environment provenance
After installing `requirements.txt`, run with the original local CSV and a fresh output directory:
```bash
python provenance.py --data /absolute/path/PhiUSIIL_Phishing_URL_Dataset.csv --out provenance/full
```
This records the CSV SHA-256, starting Git commit and working-tree status, Python,
OS, installed dependency versions, GPU/CUDA availability, and `pip freeze --all`
in `provenance.json` and `requirements-environment.txt`. Use the snapshot with
`python -m pip install -r provenance/full/requirements-environment.txt` in a compatible
Python/OS environment. This captures the current environment; it cannot recover
the environment used for the original 27 runs.

The shared training loader reports original, non-missing, and deduplicated row and
class counts (`label=0` phishing; other non-missing labels legitimate). Cleaning
removes missing URL/label values, strips URL whitespace, then keeps the first row
for each exact URL; no normalization or subsampling is applied by this command.
Signed differences compare against 235,795 original rows and 235,370 cleaned URLs
(100,520 phishing, 134,850 legitimate). Missing values, whitespace, duplicate order,
or a different dataset version can affect these counts; the checksum identifies
the exact input. URLs are processed only as strings and are never fetched.

The original CSV found locally in Downloads was verified: 235,795 original rows
(100,945 phishing, 134,850 legitimate), no missing URL/label rows, and 235,370
deduplicated URLs (100,520 phishing, 134,850 legitimate). All reference differences
are zero; deduplication removes 425 phishing rows. Local capture artifacts are in
`provenance/full/`. Input SHA-256:
`a236549cd369cd80bd478ff8e1779cbf44c58d5c3f79f7a51a1adbed7d06d1c6`. Verify the synthetic fixture and training integration with
`python -m unittest discover -s tests -v`. The historical training environment
and a new full training run remain unverified. The command captures provenance
only; use `run_all.py` separately for training. Keep datasets and per-URL outputs
out of Git; CSV files, `preds/` directories, and `provenance/` are ignored.
