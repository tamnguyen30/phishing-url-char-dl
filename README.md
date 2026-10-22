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