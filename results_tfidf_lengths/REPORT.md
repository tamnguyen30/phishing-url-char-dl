# Stripped TF-IDF length comparison

Real CSV comparison; frozen training and validation only. Test holdout was not fitted, transformed, predicted or scored.

| Stripped input | Validation phishing precision | Recall | F1 |
|---|---:|---:|---:|
| Full length | 0.939464 | 0.825049 | 0.878547 |
| First 200 characters | 0.939416 | 0.825375 | 0.878711 |
| Change (200 − full) | -0.000047 | +0.000326 | +0.000164 |

Split seed: 42; model RNG seed: 42; threshold: 0.5.
Manifest SHA-256: `96fe7538fc21c403d7df981bd905b8303c53297ee58371a68fc8d3aeb7fdf0cc`.
Dataset SHA-256: `a236549cd369cd80bd478ff8e1779cbf44c58d5c3f79f7a51a1adbed7d06d1c6`.

| Partition | Rows | Phishing | Legitimate | Filtered after stripping |
|---|---:|---:|---:|---:|
| train | 166,319 | 71,855 | 94,464 | 0 |
| validation | 35,457 | 15,330 | 20,127 | 0 |

Both runs use independent training-only vocabularies and IDF, the existing baseline hyperparameters, and the same rows and labels. No hyperparameter selection.

Exact command used (from repository root):

```bash
/Users/shanabilliot/phishing-url-char-dl/.venv/bin/python compare_tfidf_lengths.py --data /Users/shanabilliot/Downloads/PhiUSIIL_Phishing_URL_Dataset.csv --manifest private_splits/robustness-seed42.split-manifest.json --out results_tfidf_lengths --seed 42
```

See comparison.json for full estimator settings, versions, convergence diagnostics, row-assignment digests and unrounded metrics.
