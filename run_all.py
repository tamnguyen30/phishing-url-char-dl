"""Run every model x normalization x seed. Skips runs whose result file already exists.

  python run_all.py --data data/PhiUSIIL_Phishing_URL_Dataset.csv
"""
import argparse, os, subprocess, sys

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--out", default=None)
ap.add_argument("--group_by", choices=["hostname", "registrable"], default="hostname")
ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
ap.add_argument("--models", nargs="+", default=["tfidf_lr", "charcnn", "bilstm"])
ap.add_argument("--norms", nargs="+", default=["none", "strip", "host"])
a = ap.parse_args()
a.out = a.out or ("results" if a.group_by == "hostname" else "results_registrable")
if a.group_by == "registrable" and os.path.realpath(a.out) == os.path.realpath(os.path.join(os.path.dirname(__file__), "results")):
    ap.error("registrable runs require a separate output directory")

for seed in a.seeds:
    for norm in a.norms:
        for m in a.models:
            tag = f"{m}_norm{norm}_len200_emb64_seed{seed}.json"  # matches train.py defaults
            if os.path.exists(os.path.join(a.out, tag)):
                print("skip", tag); continue
            cmd = [sys.executable, "train.py", "--data", a.data, "--model", m,
                   "--group_by", a.group_by, "--normalize", norm, "--seed", str(seed), "--out", a.out]
            print(">>", " ".join(cmd), flush=True)
            subprocess.run(cmd, check=True)
