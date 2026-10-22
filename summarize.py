"""Aggregate results/*.json into results/summary.csv (mean +/- std over seeds) and print a markdown table."""
import glob, json, sys
import pandas as pd

d = sys.argv[1] if len(sys.argv) > 1 else "results"
rows = []
for f in glob.glob(f"{d}/*.json"):
    r = json.load(open(f)); a = r["args"]
    if a.get("max_rows"):  # skip debug subsamples
        continue
    rows.append(dict(model=a["model"], normalize=a["normalize"], max_len=a["max_len"], emb_dim=a["emb_dim"],
                     seed=a["seed"], **r["test"]))
if not rows:
    sys.exit(f"no result files found in {d}/")
df = pd.DataFrame(rows)
keys = ["normalize", "model", "max_len", "emb_dim"]
cols = ["precision", "recall", "f1", "auroc", "auprc"]
g = df.groupby(keys)[cols].agg(["mean", "std"])
g["n_seeds"] = df.groupby(keys).size()
g.columns = ["_".join(c) if isinstance(c, tuple) and c[1] else c[0] for c in g.columns]
g = g.reset_index().sort_values(["normalize", "model"])
g.to_csv(f"{d}/summary.csv", index=False)
print("| norm | model | n | precision | recall | F1 | AUROC |\n|---|---|---|---|---|---|---|")
for _, r in g.iterrows():
    fmt = lambda c: f"{r[c + '_mean']:.4f} ± {0 if pd.isna(r[c + '_std']) else r[c + '_std']:.4f}"
    print(f"| {r['normalize']} | {r['model']} | {int(r['n_seeds'])} | {fmt('precision')} | {fmt('recall')} | {fmt('f1')} | {fmt('auroc')} |")
