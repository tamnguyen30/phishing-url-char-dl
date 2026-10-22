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
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader, TensorDataset


# ----------------------------------------------------------------- utilities
def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def host_of(url):
    u = url if "://" in url else "http://" + url
    try:
        h = (urlparse(u).hostname or "").lower()
    except ValueError:
        return ""
    return h[4:] if h.startswith("www.") else h  # www.x.com and x.com are the same group


def strip_prefix(url):
    """Remove only the scheme (https://) and a leading 'www.'; keep host, path and query."""
    u = re.sub(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", "", url)
    return re.sub(r"^www\.", "", u, flags=re.I)


def load_and_split(args):
    df = pd.read_csv(args.data)
    df = df[[args.url_col, args.label_col]].dropna()
    df.columns = ["url", "label"]
    df["url"] = df["url"].astype(str).str.strip()
    n0 = len(df)
    df = df.drop_duplicates(subset="url").reset_index(drop=True)
    if args.max_rows:
        df = df.sample(args.max_rows, random_state=args.seed).reset_index(drop=True)
    df["y"] = (df["label"] == args.phish_label).astype(int)  # 1 = phishing
    df["host"] = df["url"].map(host_of)
    if args.normalize == "host":
        # remove path/scheme/www so the model cannot use the "has a path" shortcut
        df["url"] = df["host"]
        df = df[df["url"] != ""].reset_index(drop=True)
    elif args.normalize == "strip":
        # remove only scheme + leading www.; the path is kept
        df["url"] = df["url"].map(strip_prefix)
        df = df[df["url"] != ""].reset_index(drop=True)
    print(f"rows: {n0} -> {len(df)} after dedup | phishing rate: {df['y'].mean():.3f}")

    gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=args.seed)
    trval_idx, test_idx = next(gss.split(df, groups=df["host"]))
    trval = df.iloc[trval_idx]
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.15 / 0.85, random_state=args.seed)
    tr_i, va_i = next(gss2.split(trval, groups=trval["host"]))
    train, val, test = trval.iloc[tr_i], trval.iloc[va_i], df.iloc[test_idx]

    # leakage check
    assert not (set(train.host) & set(test.host)), "host leakage train/test"
    assert not (set(train.host) & set(val.host)), "host leakage train/val"
    assert not (set(val.host) & set(test.host)), "host leakage val/test"
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
    ap.add_argument("--out", default="results")
    ap.add_argument("--no_preds", action="store_true", help="do not save per-URL test predictions")
    args = ap.parse_args()

    set_seed(args.seed)
    train, val, test = load_and_split(args)
    test_m, extra, p_test = (run_tfidf if args.model == "tfidf_lr" else run_neural)(args, train, val, test)
    print("TEST (host-disjoint):", {k: round(v, 4) for k, v in test_m.items()})

    os.makedirs(args.out, exist_ok=True)
    tag = f"{args.model}_norm{args.normalize}_len{args.max_len}_emb{args.emb_dim}_seed{args.seed}"
    with open(os.path.join(args.out, tag + ".json"), "w") as f:
        json.dump(dict(args=vars(args), test=test_m, extra=extra), f, indent=2)
    if not args.no_preds:  # for error analysis; results/preds/ is gitignored (derived from the dataset)
        pdir = os.path.join(args.out, "preds"); os.makedirs(pdir, exist_ok=True)
        test.assign(p=p_test)[["url", "host", "y", "p"]].to_csv(os.path.join(pdir, tag + ".csv"), index=False)
