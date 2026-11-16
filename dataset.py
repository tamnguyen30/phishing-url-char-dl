"""Shared CSV cleaning and class mapping; URL values are never fetched."""
import pandas as pd


def class_counts(df, label_col, phish_label):
    missing = int(df[label_col].isna().sum())
    phishing = int((df[label_col] == phish_label).sum())
    return dict(rows=len(df), phishing=phishing,
                legitimate=len(df) - missing - phishing, missing_labels=missing)


def load_dataset(path, url_col="URL", label_col="label", phish_label=0):
    # Force URL strings, including numeric-looking values; preserve pandas' NA policy.
    df = pd.read_csv(path, dtype={url_col: str})
    counts = {"original": class_counts(df, label_col, phish_label)}
    df = df[[url_col, label_col]].dropna()
    df.columns = ["url", "label"]
    df["url"] = df["url"].astype(str).str.strip()
    counts["non_missing"] = class_counts(df, "label", phish_label)
    df = df.drop_duplicates(subset="url").reset_index(drop=True)
    df["y"] = (df["label"] == phish_label).astype(int)
    counts["deduplicated"] = class_counts(df, "label", phish_label)
    return df, counts
