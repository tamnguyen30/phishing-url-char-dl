# Pinned Public Suffix List

`public_suffix_list.dat` is copied byte-for-byte from the bundled snapshot in
`tldextract==5.3.0` (package resource `.tld_set_snapshot`). Its upstream header records:

- VERSION: `2025-04-07_15-51-09_UTC`
- COMMIT: `5fc8d12c1624ddbc56a80e944aca151a8ee466a1`
- SHA-256: `b69315c085d53972724b8f2df111ffc329b0c84fe0a47d62c8c91655cc774a38`

Upstream source: <https://publicsuffix.org/list/public_suffix_list.dat>.
The snapshot retains its Mozilla Public License 2.0 notice. Updates must be deliberate:
replace the committed snapshot, update this metadata and its test, and rerun the audit.
The runtime uses this local file only, disables caching and package-snapshot fallback,
and includes both ICANN and PRIVATE rules. It never updates the list automatically.
See [tldextract configuration](https://github.com/john-kurkowski/tldextract#advanced-usage).
