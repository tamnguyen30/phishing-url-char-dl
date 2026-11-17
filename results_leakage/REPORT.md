# Real-data split leakage audit

Command:

```bash
.venv/bin/python audit_splits.py --data /Users/shanabilliot/Downloads/PhiUSIIL_Phishing_URL_Dataset.csv --out results_leakage --seeds 42 43 44
```

Input SHA-256: `a236549cd369cd80bd478ff8e1779cbf44c58d5c3f79f7a51a1adbed7d06d1c6`. Starting commit: `b99c48d40fd9a0972b8b79ecc40dc06566821bf5`;
working-tree status is recorded in `audit.json` because this capture ran during implementation.
Environment: `requirements-environment.txt`; parser and snapshot details: `audit.json`.

235,795 original rows; 235,370 cleaned URLs (100,520 phishing, 134,850 legitimate).
Grouping status: 233,998 rows with a registrable domain, 606 IP rows, 766 rows using
full-host fallback, and 0 malformed rows. No rows were excluded from the exact audit. Unique grouping keys by status:
197,538 registrable domains, 157 IPs,
14 full-host fallback groups, and 0 malformed groups.

The audit uses original deduplicated URL strings, before model normalization.
Every registrable split has zero shared registrable domains **and** zero shared fallback keys.
The hostname workflow was independently compared with `b99c48d:train.py` on the real CSV
for all three seeds and all three normalizations; row membership, order, labels and
normalized URLs matched in all nine configurations. `run_all.py` skipped all 27 original runs.

## Partition sizes

“Domains” counts actual registrable domains; “all groups” also includes fallback keys.
Hostname counts are available in `audit.json`. Each seed partitions all 235,370 rows.

| Grouping | Seed | Partition | URLs | Phishing | Legitimate | Domains | All groups |
|---|---:|---|---:|---:|---:|---:|---:|
| hostname | 42 | train | 164,421 | 70,120 | 94,301 | 139,097 | 139,214 |
| hostname | 42 | validation | 35,531 | 15,242 | 20,289 | 30,496 | 30,529 |
| hostname | 42 | test | 35,418 | 15,158 | 20,260 | 30,565 | 30,586 |
| hostname | 43 | train | 164,960 | 70,618 | 94,342 | 139,115 | 139,229 |
| hostname | 43 | validation | 35,153 | 14,949 | 20,204 | 30,453 | 30,484 |
| hostname | 43 | test | 35,257 | 14,953 | 20,304 | 30,482 | 30,508 |
| hostname | 44 | train | 164,164 | 69,676 | 94,488 | 138,997 | 139,128 |
| hostname | 44 | validation | 35,861 | 15,726 | 20,135 | 30,559 | 30,581 |
| hostname | 44 | test | 35,345 | 15,118 | 20,227 | 30,491 | 30,509 |
| registrable | 42 | train | 166,319 | 71,855 | 94,464 | 138,286 | 138,395 |
| registrable | 42 | validation | 35,457 | 15,330 | 20,127 | 29,626 | 29,657 |
| registrable | 42 | test | 33,594 | 13,335 | 20,259 | 29,626 | 29,657 |
| registrable | 43 | train | 166,962 | 72,860 | 94,102 | 138,276 | 138,395 |
| registrable | 43 | validation | 34,974 | 14,623 | 20,351 | 29,628 | 29,657 |
| registrable | 43 | test | 33,434 | 13,037 | 20,397 | 29,634 | 29,657 |
| registrable | 44 | train | 166,719 | 72,222 | 94,497 | 138,275 | 138,395 |
| registrable | 44 | validation | 33,479 | 13,236 | 20,243 | 29,631 | 29,657 |
| registrable | 44 | test | 35,172 | 15,062 | 20,110 | 29,632 | 29,657 |

## Cross-partition audit

Each cell counts distinct shared keys, not URL pairs. Path and query templates
are possible similarities across different domain grouping keys, not confirmed leakage.
Exact counts remove only scheme/leading `www.` and lowercase the hostname.
See the root README for complete rules. `audit.json` also records contributing URL
counts on each side of every comparison and partition-specific fallback counts.

| Grouping | Seed | Pair | Shared domains | Exact keys | Possible path templates | Possible query templates |
|---|---:|---|---:|---:|---:|---:|
| hostname | 42 | train/validation | 1,270 | 0 | 803 | 129 |
| hostname | 42 | train/test | 1,265 | 0 | 730 | 139 |
| hostname | 42 | validation/test | 663 | 0 | 388 | 85 |
| hostname | 43 | train/validation | 1,211 | 0 | 864 | 135 |
| hostname | 43 | train/test | 1,215 | 0 | 641 | 133 |
| hostname | 43 | validation/test | 621 | 0 | 432 | 84 |
| hostname | 44 | train/validation | 1,248 | 0 | 732 | 142 |
| hostname | 44 | train/test | 1,181 | 0 | 517 | 137 |
| hostname | 44 | validation/test | 620 | 0 | 317 | 91 |
| registrable | 42 | train/validation | 0 | 0 | 661 | 137 |
| registrable | 42 | train/test | 0 | 0 | 437 | 131 |
| registrable | 42 | validation/test | 0 | 0 | 223 | 87 |
| registrable | 43 | train/validation | 0 | 0 | 687 | 130 |
| registrable | 43 | train/test | 0 | 0 | 405 | 123 |
| registrable | 43 | validation/test | 0 | 0 | 207 | 76 |
| registrable | 44 | train/validation | 0 | 0 | 515 | 130 |
| registrable | 44 | train/test | 0 | 1 | 479 | 130 |
| registrable | 44 | validation/test | 0 | 0 | 237 | 81 |

The seed-44 registrable train/test split has one normalized exact key, involving
2 training URLs and 1 test URL. Aggregate inspection found the training URLs use
full-host fallback for a suffix-only host, whereas the test URL has leading `www.`
and a registrable domain. Removing `www.` erases that PSL boundary. Thus zero
shared domain groups does not guarantee zero matches under this URL normalization.
The report also records shared hostname counts; that same boundary can cross the
historical leading-`www.` hostname equivalence. No source URLs or keys are published.
All hostname split pairs and the remaining registrable split pairs have zero
normalized exact matches.

Verification: six tests pass (offline suffix rules, private suffixes, IP/malformed
fallbacks, original training compatibility, missing-template exclusion and audit semantics);
`pip check`, byte compilation and `git diff --check` pass. Original result files
are unchanged. No full model retraining or network/content inspection of dataset
URLs was performed. Two 500-row TF-IDF smoke tests (one per grouping method) passed,
saved outside the original results directory with `--no_preds`. The smoke tests
exercise the training/output path; their metrics are not full-dataset results.
Possible similarities do not establish model performance impact.
