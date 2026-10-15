#!/usr/bin/env python
"""
BanglaNLP - Bangla text classification & sentiment analysis.

Examples
--------
# Classical ML only (fast, CPU)
python main.py --data data/sample_bangla_sentiment.csv --models ml

# ML + BiLSTM/TextCNN
python main.py --data data/your_data.csv --models ml dl

# Everything, including BanglaBERT fine-tuning (GPU recommended)
python main.py --data data/your_data.csv --models ml dl transformer

# Pre-split files and numeric labels (e.g. SentNoB-style data)
python main.py --train train.csv --val val.csv --test test.csv \
               --text-col Data --label-col Label --label-map "0:neutral,1:positive,2:negative"
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np

from src.data_utils import (
    attach_label_ids,
    build_label_names,
    check_overlap,
    describe_dataset,
    load_dataset,
    parse_label_map,
    split_dataset,
)
from src.evaluation import plot_class_distribution, summarize_results
from src.preprocessing import BanglaPreprocessor, PreprocessConfig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="BanglaNLP: Bangla text classification")
    data = p.add_argument_group("data")
    data.add_argument("--data", help="single file; will be split into train/val/test")
    data.add_argument("--train")
    data.add_argument("--val")
    data.add_argument("--test")
    data.add_argument("--text-col", default="text")
    data.add_argument("--label-col", default="label")
    data.add_argument("--label-map", help='e.g. "0:neutral,1:positive,2:negative"')
    data.add_argument("--test-size", type=float, default=0.1)
    data.add_argument("--val-size", type=float, default=0.1)

    p.add_argument("--models", nargs="+", default=["ml", "dl"],
                   choices=["ml", "dl", "transformer"])
    p.add_argument("--output-dir", default="results")
    p.add_argument("--seed", type=int, default=42)

    pre = p.add_argument_group("preprocessing (ML / DL)")
    pre.add_argument("--remove-stopwords", action="store_true")
    pre.add_argument("--stem", action="store_true")

    ml = p.add_argument_group("classical ML")
    ml.add_argument("--features", nargs="+", default=["bow", "tfidf_word", "tfidf_word_char"])
    ml.add_argument("--classifiers", nargs="+",
                    default=["logreg", "linear_svm", "naive_bayes", "random_forest"])

    dl = p.add_argument_group("deep learning")
    dl.add_argument("--dl-models", nargs="+", default=["bilstm", "cnn"])
    dl.add_argument("--dl-epochs", type=int, default=15)
    dl.add_argument("--dl-batch-size", type=int, default=64)
    dl.add_argument("--embeddings", help="fastText .vec file, e.g. cc.bn.300.vec")
    dl.add_argument("--embed-dim", type=int, default=300)

    tf = p.add_argument_group("transformer")
    tf.add_argument("--transformer-model", default="csebuetnlp/banglabert")
    tf.add_argument("--tf-epochs", type=int, default=4)
    tf.add_argument("--tf-batch-size", type=int, default=16)
    tf.add_argument("--tf-lr", type=float, default=2e-5)
    tf.add_argument("--max-length", type=int, default=128)
    tf.add_argument("--no-bangla-normalizer", action="store_true")
    return p.parse_args()


def load_splits(args):
    label_map = parse_label_map(args.label_map)
    kw = dict(text_col=args.text_col, label_col=args.label_col, label_map=label_map)
    if args.data:
        df = load_dataset(args.data, **kw)
        return split_dataset(df, args.test_size, args.val_size, args.seed)
    if not (args.train and args.test):
        raise SystemExit("Provide --data, or --train and --test (and optionally --val).")
    train, test = load_dataset(args.train, **kw), load_dataset(args.test, **kw)
    if args.val:
        val = load_dataset(args.val, **kw)
    else:
        from sklearn.model_selection import train_test_split
        train, val = train_test_split(train, test_size=args.val_size,
                                      stratify=train["label"], random_state=args.seed)
        train, val = train.reset_index(drop=True), val.reset_index(drop=True)
    check_overlap(train, val, "val")
    check_overlap(train, test, "test")
    return train, val, test


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ---------------- data ----------------
    train, val, test = load_splits(args)
    label_names = build_label_names(train)
    train, val, test = (attach_label_ids(d, label_names) for d in (train, val, test))
    print(f"[data] labels: {label_names}")
    for name, d in (("train", train), ("val", val), ("test", test)):
        describe_dataset(name, d)
    plot_class_distribution(train, val, test, out)

    pp_cfg = PreprocessConfig(remove_stopwords=args.remove_stopwords, stem=args.stem)
    pp = BanglaPreprocessor(pp_cfg)
    print("\n[preprocess] examples:")
    for t in train["text"].head(3):
        print(f"  RAW   : {t}\n  TOKENS: {pp.process(t)}")
    print()

    # ---------------- models ----------------
    results = []
    if "ml" in args.models:
        from src.ml_models import run_ml_experiments
        min_df = 2 if len(train) >= 2000 else 1
        results += run_ml_experiments(train, val, test, label_names, out,
                                      features=args.features, classifiers=args.classifiers,
                                      min_df=min_df, seed=args.seed,
                                      remove_stopwords=args.remove_stopwords, stem=args.stem)
    if "dl" in args.models:
        from src.dl_models import run_dl_experiments
        results += run_dl_experiments(train, val, test, label_names, out,
                                      model_types=args.dl_models, preprocess_cfg=pp_cfg,
                                      epochs=args.dl_epochs, batch_size=args.dl_batch_size,
                                      embeddings_path=args.embeddings,
                                      embed_dim=args.embed_dim, seed=args.seed)
    if "transformer" in args.models:
        from src.transformer_model import run_transformer_experiment
        results += run_transformer_experiment(
            train, val, test, label_names, out, model_name=args.transformer_model,
            max_length=args.max_length, epochs=args.tf_epochs, batch_size=args.tf_batch_size,
            lr=args.tf_lr, use_bangla_normalizer=not args.no_bangla_normalizer, seed=args.seed)

    summarize_results(results, out)
    (out / "run_config.json").write_text(json.dumps(
        {**vars(args), "label_names": label_names,
         "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=2), encoding="utf-8")
    print(f"\nSaved reports, confusion matrices, error files and models to: {out.resolve()}")


if __name__ == "__main__":
    main()
