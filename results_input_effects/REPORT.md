# URL input diagnostics

Scope: pooled frozen training + validation only.

| Input | Class | Denominator | https://www. count (%) | Nonempty path count (%) | Query count (%) |
|---|---|---:|---:|---:|---:|
| raw | phishing | 87,185 | 2,124 (2.436%) | 52,340 (60.033%) | 5,331 (6.115%) |
| raw | legitimate | 114,591 | 114,591 (100.000%) | 0 (0.000%) | 0 (0.000%) |
| stripped | phishing | 87,185 | 0 (0.000%) | 52,340 (60.033%) | 5,331 (6.115%) |
| stripped | legitimate | 114,591 | 0 (0.000%) | 0 (0.000%) | 0 (0.000%) |

| Input | Class | Denominator | >200 characters count (%) | Median characters removed (affected only) |
|---|---|---:|---:|---:|
| raw | phishing | 87,185 | 1,306 (1.498%) | 63.5 |
| raw | legitimate | 114,591 | 0 (0.000%) | N/A |
| stripped | phishing | 87,185 | 1,160 (1.331%) | 77 |
| stripped | legitimate | 114,591 | 0 (0.000%) | N/A |
| host-only | phishing | 87,185 | 0 (0.000%) | N/A |
| host-only | legitimate | 114,591 | 0 (0.000%) | N/A |

Full-dataset historical check: 1,427 of 235,370 deduplicated raw URLs exceed 200 characters; 1,427 phishing, 0 legitimate. Matches report: True.

Rows filtered after preprocessing (input/class/count): raw/phishing/0, raw/legitimate/0, stripped/phishing/0, stripped/legitimate/0, host-only/phishing/0, host-only/legitimate/0.
Formatting parse failures: 0.
No model fitting, model selection, predictions or scores.
