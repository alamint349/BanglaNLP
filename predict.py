#!/usr/bin/env python
"""
Predict sentiment / class for new Bangla text with any saved model.

python predict.py --model results/models/tfidf_word_char__logreg.joblib --text "খাবারটা দারুণ ছিল"
python predict.py --model results/models/dl__bilstm.pt --file sentences.txt
python predict.py --model results/models/transformer__banglabert/best      # interactive
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np

Prediction = Tuple[str, Optional[float]]


def load_predictor(model_path: str) -> Callable[[List[str]], List[Prediction]]:
    path = Path(model_path)

    if path.suffix == ".joblib":
        import joblib
        bundle = joblib.load(path)
        pipe, labels = bundle["pipeline"], bundle["label_names"]

        def predict(texts):
            if hasattr(pipe, "predict_proba"):
                probs = pipe.predict_proba(texts)
                return [(labels[i], float(p[i])) for p, i in zip(probs, probs.argmax(1))]
            return [(labels[i], None) for i in pipe.predict(texts)]   # e.g. LinearSVC
        return predict

    if path.suffix == ".pt":
        from src.dl_models import load_dl_model, predict_dl
        from src.preprocessing import BanglaPreprocessor, PreprocessConfig
        model, vocab, cfg, labels, pp_cfg = load_dl_model(path)
        pp = BanglaPreprocessor(PreprocessConfig(**pp_cfg))

        def predict(texts):
            probs = predict_dl(model, vocab, pp.batch_tokens(texts), cfg)
            return [(labels[i], float(p[i])) for p, i in zip(probs, probs.argmax(1))]
        return predict

    if path.is_dir():
        from src.transformer_model import load_transformer, predict_transformer
        model, tokenizer, normalize, meta = load_transformer(str(path))
        labels = meta["label_names"]

        def predict(texts):
            probs = predict_transformer(model, tokenizer, texts, normalize, meta["max_length"])
            return [(labels[i], float(p[i])) for p, i in zip(probs, probs.argmax(1))]
        return predict

    raise ValueError(f"Unrecognised model path: {path}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--text", nargs="+", help="one or more sentences")
    p.add_argument("--file", help="UTF-8 text file, one sentence per line")
    args = p.parse_args()

    predict = load_predictor(args.model)
    texts = args.text or []
    if args.file:
        texts += [l.strip() for l in Path(args.file).read_text(encoding="utf-8").splitlines()
                  if l.strip()]

    def show(batch):
        for t, (label, conf) in zip(batch, predict(batch)):
            conf_s = f" ({conf:.2%})" if conf is not None else ""
            print(f"{label:>10}{conf_s:>10}  |  {t}")

    if texts:
        show(texts)
    else:
        print("Interactive mode - type a Bangla sentence (empty line to quit)")
        while (line := input("> ").strip()):
            show([line])


if __name__ == "__main__":
    main()
