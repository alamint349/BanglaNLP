"""Dataset loading, de-duplication, label encoding and stratified splitting."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from .preprocessing import normalize_unicode


def parse_label_map(spec: Optional[str]) -> Optional[Dict[str, str]]:
    """'0:neutral,1:positive,2:negative' -> {'0': 'neutral', ...}"""
    if not spec:
        return None
    pairs = [p.split(":", 1) for p in spec.split(",") if p.strip()]
    return {k.strip(): v.strip().lower() for k, v in pairs}


def load_dataset(path: str, text_col: str = "text", label_col: str = "label",
                 label_map: Optional[Dict[str, str]] = None) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() in {".tsv", ".txt"}:
        df = pd.read_csv(path, sep="\t", encoding="utf-8")
    elif path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(path)
    elif path.suffix.lower() in {".json", ".jsonl"}:
        df = pd.read_json(path, lines=path.suffix.lower() == ".jsonl")
    else:
        df = pd.read_csv(path, encoding="utf-8")

    missing = {text_col, label_col} - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing column(s) {missing}. Found: {list(df.columns)}")

    df = df[[text_col, label_col]].rename(columns={text_col: "text", label_col: "label"})
    df = df.dropna()
    df["text"] = df["text"].astype(str).map(normalize_unicode).str.strip()
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    if label_map:
        df["label"] = df["label"].map(lambda x: label_map.get(x, x))
    df = df[df["text"].str.len() > 0]

    before = len(df)
    df = df.drop_duplicates(subset="text").reset_index(drop=True)
    if before - len(df):
        print(f"[data] {path.name}: removed {before - len(df)} duplicate texts")
    return df


def split_dataset(df: pd.DataFrame, test_size: float = 0.1, val_size: float = 0.1,
                  seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_val, test = train_test_split(df, test_size=test_size,
                                       stratify=df["label"], random_state=seed)
    rel_val = val_size / (1.0 - test_size)
    train, val = train_test_split(train_val, test_size=rel_val,
                                  stratify=train_val["label"], random_state=seed)
    return (train.reset_index(drop=True), val.reset_index(drop=True),
            test.reset_index(drop=True))


def check_overlap(train: pd.DataFrame, other: pd.DataFrame, name: str) -> None:
    n = len(set(train["text"]) & set(other["text"]))
    if n:
        print(f"[data] WARNING: {n} texts in '{name}' also appear in train (leakage)")


def build_label_names(train: pd.DataFrame) -> List[str]:
    return sorted(train["label"].unique().tolist())


def attach_label_ids(df: pd.DataFrame, label_names: List[str]) -> pd.DataFrame:
    label2id = {l: i for i, l in enumerate(label_names)}
    unknown = set(df["label"]) - set(label2id)
    if unknown:
        raise ValueError(f"Labels {unknown} not seen in training data")
    df = df.copy()
    df["label_id"] = df["label"].map(label2id).astype(int)
    return df


def describe_dataset(name: str, df: pd.DataFrame) -> None:
    lengths = df["text"].str.split().str.len()
    dist = df["label"].value_counts().to_dict()
    print(f"[data] {name:<5} n={len(df):>6} | words/text mean={lengths.mean():.1f} "
          f"max={lengths.max()} | classes={dist}")
