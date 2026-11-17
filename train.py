"""Phishing URL detection: TF-IDF baseline vs. character-level neural models.

Usage examples:
  python train.py --data data/PhiUSIIL_Phishing_URL_Dataset.csv --model tfidf_lr
  python train.py --data data/PhiUSIIL_Phishing_URL_Dataset.csv --model charcnn
  python train.py --data data/PhiUSIIL_Phishing_URL_Dataset.csv --model bilstm
  python train.py --data ... --model charcnn --max_len 100 --emb_dim 32   # ablation
  python train.py --data ... --model charcnn --normalize strip            # drop scheme + leading www., keep path
  python train.py --data ... --model charcnn --normalize host             # hostname-only URLs

Key design choices (to avoid leakage):
  * exact-duplicate URLs are dropped
  * train/val/test are split by HOSTNAME (GroupShuffleSplit), so no host appears in two splits
"""
import argparse, json, os, random, re, time
from collections import Counter

import numpy as np
from dataset import load_dataset
import torch
import torch.nn as nn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,
                             precision_score, recall_score, roc_auc_score)
from splitting import host_of, domain_of, split_frame, parser_metadata
from torch.utils.data import DataLoader, TensorDataset


# ----------------------------------------------------------------- utilities
def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def strip_prefix(url):
    """Remove only the scheme (https://) and a leading 'www.'; keep host, path and query."""
    u = re.sub(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", "", url)
    return re.sub(r"^www\.", "", u, flags=re.I)


def load_and_split(args):
    df, counts = load_dataset(args.data, args.url_col, args.label_col, args.phish_label)
    manifest_path = getattr(args, "split_manifest", None)
    if manifest_path:
        from frozen_splits import load_manifest
        if args.max_rows:
            raise ValueError("split manifest cannot be combined with --max_rows")
        if getattr(args, "group_by", "hostname") != "registrable":
            raise ValueError("split manifest requires --group_by registrable")
        assignments, manifest = load_manifest(manifest_path, args.data, df,
            args.url_col, args.label_col, args.phish_label)
        df["partition"] = assignments
        from frozen_splits import url_hash
        df["url_sha256"] = df.url.map(url_hash)
        print("frozen split seed:", manifest["seed"], "|", manifest["holdout_type"])
    n0 = counts["non_missing"]["rows"]
    if args.max_rows:
        df = df.sample(args.max_rows, random_state=args.seed).reset_index(drop=True)
    df["host"] = df["url"].map(host_of)
    grouping = getattr(args, "group_by", "hostname")
    group_col = "host"
    if grouping == "registrable":
        parsed = df["url"].map(domain_of)
        df["domain"] = parsed.map(lambda item: item[0])
        df["domain_status"] = parsed.map(lambda item: item[1])
        print("domain parser:", parser_metadata())
        print("domain grouping status:", parsed.map(lambda item: item[1]).value_counts().to_dict())
        group_col = "domain"
    before_preprocessing = (df.partition.value_counts().to_dict() if manifest_path else {"all": len(df)})
    if args.normalize == "host":
        # remove path/scheme/www so the model cannot use the "has a path" shortcut
        df["url"] = df["host"]
        df = df[df["url"] != ""].reset_index(drop=True)
    elif args.normalize == "strip":
        # remove only scheme + leading www.; the path is kept
        df["url"] = df["url"].map(strip_prefix)
        df = df[df["url"] != ""].reset_index(drop=True)
    print(f"rows: {n0} -> {len(df)} after dedup | phishing rate: {df['y'].mean():.3f}")

    if manifest_path:
        train, val, test = (df[df.partition == name].copy() for name in ("train", "validation", "test"))
        filtered = {name: before_preprocessing.get(name, 0) - len(part)
                    for name, part in zip(("train", "validation", "test"), (train, val, test))}
    else:
        train, val, test = split_frame(df, args.seed, group_col)
        filtered = {"all": before_preprocessing["all"] - len(df)}
    print("rows filtered after preprocessing:", filtered)
    args.preprocessing_filtered = filtered
    for n, d in [("train", train), ("val", val), ("test", test)]:
        print(f"{n}: {len(d)} urls, {d.host.nunique()} hosts, phishing rate {d.y.mean():.3f}")
    return train, val, test


def metrics(y, p, thr=0.5):
    pred = (p >= thr).astype(int)
    return dict(accuracy=accuracy_score(y, pred), precision=precision_score(y, pred, zero_division=0),
                recall=recall_score(y, pred), f1=f1_score(y, pred),
                auroc=roc_auc_score(y, p), auprc=average_precision_score(y, p))


# ----------------------------------------------------------------- encoding
# printable ASCII -> 2..96, unknown -> 1, pad -> 0
def encode(urls, max_len):
    out = np.zeros((len(urls), max_len), dtype=np.int64)
    for i, u in enumerate(urls):
        for j, ch in enumerate(u[:max_len]):
            o = ord(ch)
            out[i, j] = o - 30 if 32 <= o <= 126 else 1
    return out

VOCAB = 97


# ----------------------------------------------------------------- models
class CharCNN(nn.Module):
    def __init__(self, emb_dim=64, n_filters=128, kernels=(3, 5, 7), dropout=0.3):
        super().__init__()
        self.emb = nn.Embedding(VOCAB, emb_dim, padding_idx=0)
        self.convs = nn.ModuleList([nn.Conv1d(emb_dim, n_filters, k, padding=k // 2) for k in kernels])
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(n_filters * len(kernels), 1)

    def forward(self, x):
        h = self.emb(x).transpose(1, 2)                       # B, E, L
        feats = [torch.relu(c(h)).max(dim=2).values for c in self.convs]
        return self.fc(self.drop(torch.cat(feats, dim=1))).squeeze(1)


class BiLSTM(nn.Module):
    def __init__(self, emb_dim=64, hidden=128, dropout=0.3):
        super().__init__()
        self.emb = nn.Embedding(VOCAB, emb_dim, padding_idx=0)
        self.rnn = nn.LSTM(emb_dim, hidden, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(2 * hidden, 1)

    def forward(self, x):
        h, _ = self.rnn(self.emb(x))
        return self.fc(self.drop(h.max(dim=1).values)).squeeze(1)


def build(name, a):
    if name == "charcnn":
        return CharCNN(a.emb_dim)
    if name == "bilstm":
        return BiLSTM(a.emb_dim)
    raise ValueError(name)


# ----------------------------------------------------------------- training
@torch.no_grad()
def predict(model, loader, dev):
    model.eval(); ps = []
    for (xb,) in loader:
        ps.append(torch.sigmoid(model(xb.to(dev))).cpu().numpy())
    return np.concatenate(ps)


def run_neural(args, train, val, test):
    dev = torch.device("cuda" if torch.cuda.is_available() else
                       "mps" if torch.backends.mps.is_available() else "cpu")
    print("device:", dev)
    X = {k: torch.from_numpy(encode(d.url.tolist(), args.max_len)) for k, d in
         [("train", train), ("val", val), ("test", test)]}
    ytr = torch.tensor(train.y.values, dtype=torch.float32)
    tl = DataLoader(TensorDataset(X["train"], ytr), batch_size=args.bs, shuffle=True)
    vl = DataLoader(TensorDataset(X["val"]), batch_size=512)
    tel = DataLoader(TensorDataset(X["test"]), batch_size=512)

    model = build(args.model, args).to(dev)
    print(f"params: {sum(p.numel() for p in model.parameters()):,}")
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    lossf = nn.BCEWithLogitsLoss()
    best, best_state, bad, curve = -1, None, 0, []
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        model.train(); tot = 0
        for xb, yb in tl:
            xb, yb = xb.to(dev), yb.to(dev)
            opt.zero_grad(); loss = lossf(model(xb), yb); loss.backward(); opt.step()
            tot += loss.item() * len(xb)
        vm = metrics(val.y.values, predict(model, vl, dev))
        curve.append(dict(epoch=ep, train_loss=tot / len(train), **{f"val_{k}": v for k, v in vm.items()}))
        print(f"ep {ep:02d} loss {tot/len(train):.4f} val_auroc {vm['auroc']:.4f} val_f1 {vm['f1']:.4f}")
        if vm["auroc"] > best:
            best, bad = vm["auroc"], 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}  # in memory only
        else:
            bad += 1
            if bad >= args.patience:
                break
    train_time = time.time() - t0
    model.load_state_dict(best_state)
    t1 = time.time(); p = predict(model, tel, dev); infer_time = time.time() - t1
    return metrics(test.y.values, p), dict(curve=curve, train_time_s=train_time, test_infer_time_s=infer_time), p


def run_tfidf(args, train, val, test):
    t0 = time.time()
    vec = TfidfVectorizer(analyzer="char", ngram_range=(2, 5), min_df=5, sublinear_tf=True,
                          max_features=500_000, lowercase=False)
    Xtr = vec.fit_transform(train.url)
    clf = LogisticRegression(max_iter=1000, C=args.C, n_jobs=-1).fit(Xtr, train.y)
    extra = dict(train_time_s=time.time() - t0,
                 val=metrics(val.y.values, clf.predict_proba(vec.transform(val.url))[:, 1]))
    p = clf.predict_proba(vec.transform(test.url))[:, 1]
    return metrics(test.y.values, p), extra, p


# ----------------------------------------------------------------- main
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--url_col", default="URL")
    ap.add_argument("--label_col", default="label")
    ap.add_argument("--phish_label", type=int, default=0,
                    help="value in label column meaning phishing (PhiUSIIL: 0; VERIFY for your file)")
    ap.add_argument("--model", choices=["tfidf_lr", "charcnn", "bilstm"], required=True)
    ap.add_argument("--normalize", choices=["none", "strip", "host"], default="none",
                    help="none: raw URL | strip: drop scheme and leading www., keep path | "
                         "host: replace each URL by its hostname (drops the path too)")
    ap.add_argument("--max_len", type=int, default=200)
    ap.add_argument("--emb_dim", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--C", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_rows", type=int, default=0, help="subsample for quick debugging")
    ap.add_argument("--group_by", choices=["hostname", "registrable"], default="hostname")
    ap.add_argument("--split_manifest", help="private frozen assignments; requires registrable grouping; seed controls model initialization only")
    ap.add_argument("--split_only", action="store_true", help="validate splits and preprocessing, then exit without training or scoring")
    ap.add_argument("--out", default=None)
    ap.add_argument("--no_preds", action="store_true", help="do not save per-URL test predictions")
    args = ap.parse_args()

    args.out = args.out or ("results" if args.group_by == "hostname" else "results_registrable")
    if args.group_by == "registrable" and os.path.realpath(args.out) == os.path.realpath(os.path.join(os.path.dirname(__file__), "results")):
        ap.error("registrable runs require a separate output directory")
    set_seed(args.seed)
    try:
        train, val, test = load_and_split(args)
    except (ValueError, OSError) as exc:
        ap.error(str(exc))
    if args.split_only:
        raise SystemExit(0)
    test_m, extra, p_test = (run_tfidf if args.model == "tfidf_lr" else run_neural)(args, train, val, test)
    split_label = "host-disjoint" if args.group_by == "hostname" else "registrable-group-disjoint"
    print(f"TEST ({split_label}):", {k: round(v, 4) for k, v in test_m.items()})

    os.makedirs(args.out, exist_ok=True)
    tag = f"{args.model}_norm{args.normalize}_len{args.max_len}_emb{args.emb_dim}_seed{args.seed}"
    if args.split_manifest:
        from provenance import sha256
        from pathlib import Path
        tag += "_split" + sha256(Path(args.split_manifest))[:12]
    with open(os.path.join(args.out, tag + ".json"), "w") as f:
        json.dump(dict(args=vars(args), test=test_m, extra=extra,
                       grouping_metadata=dict(**parser_metadata(), partitions={
                           name: dict(Counter(part.domain_status))
                           for name, part in (("train", train), ("validation", val), ("test", test))
                       }) if args.group_by == "registrable" else None), f, indent=2)
    if not args.no_preds:  # for error analysis; results/preds/ is gitignored (derived from the dataset)
        pdir = os.path.join(args.out, "preds"); os.makedirs(pdir, exist_ok=True)
        test.assign(p=p_test)[["url", "host", "y", "p"]].to_csv(os.path.join(pdir, tag + ".csv"), index=False)
