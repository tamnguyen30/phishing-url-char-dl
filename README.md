# Deep Learning for Phishing URL Detection Using Character-Level Features

COSC 50523 course project (preliminary report code).

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
python train.py --data data/PhiUSIIL_Phishing_URL_Dataset.csv --model tfidf_lr   # baseline
python train.py --data data/PhiUSIIL_Phishing_URL_Dataset.csv --model charcnn
python train.py --data data/PhiUSIIL_Phishing_URL_Dataset.csv --model bilstm
# hostname-only version (removes the path shortcut)
python train.py --data data/PhiUSIIL_Phishing_URL_Dataset.csv --model charcnn --normalize host
# quick smoke test
python train.py --data data/PhiUSIIL_Phishing_URL_Dataset.csv --model charcnn --max_rows 5000 --epochs 2
```
Results are written to `results/*.json` (test metrics, validation curve, timing, full config).

## Protocol
- Exact duplicate URLs removed.
- 70/15/15 train/val/test split grouped by hostname (no host appears in two splits; asserted in code).
- Model selection on validation AUROC; test set touched once per run.
- Seed fixed (default 42).

## Not committed
Datasets, checkpoints, and weights (see `.gitignore`).
