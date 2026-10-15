"""
Fine-tune a pretrained transformer for Bangla text classification.

Default model: csebuetnlp/banglabert (ELECTRA pretrained on Bangla text).
Alternatives:  sagorsarker/bangla-bert-base, google/muril-base-cased,
               xlm-roberta-base (multilingual).

BanglaBERT expects text normalised with the authors' normaliser:
    pip install git+https://github.com/csebuetnlp/normalizer
If it is not installed, the built-in `light_normalize` is used on its own.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Callable, List, Optional, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

from .evaluation import compute_metrics, evaluate_and_save
from .preprocessing import light_normalize


def get_text_normalizer(use_bangla_normalizer: bool = True) -> Callable[[str], str]:
    bn_normalize = None
    if use_bangla_normalizer:
        try:
            from normalizer import normalize as bn_normalize  # csebuetnlp/normalizer
        except ImportError:
            print("[transformer] csebuetnlp 'normalizer' not installed -> using built-in "
                  "light normalisation only")

    def _normalize(text: str) -> str:
        text = light_normalize(text)
        return bn_normalize(text) if bn_normalize else text

    return _normalize


class EncodedTextDataset(Dataset):
    def __init__(self, texts: Sequence[str], labels: Optional[Sequence[int]], tokenizer,
                 max_length: int):
        self.encodings = tokenizer(list(texts), truncation=True, max_length=max_length)
        self.labels = list(labels) if labels is not None else None

    def __len__(self) -> int:
        return len(self.encodings["input_ids"])

    def __getitem__(self, i):
        item = {k: v[i] for k, v in self.encodings.items()}
        if self.labels is not None:
            item["labels"] = int(self.labels[i])
        return item


class WeightedTrainer(Trainer):
    """Trainer with class-weighted cross-entropy for imbalanced sentiment data."""

    def __init__(self, *args, class_weights: Optional[torch.Tensor] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        weight = self.class_weights.to(logits.device) if self.class_weights is not None else None
        loss = nn.functional.cross_entropy(logits, labels, weight=weight)
        return (loss, outputs) if return_outputs else loss


def _hf_metrics(eval_pred):
    logits, labels = eval_pred.predictions, eval_pred.label_ids
    if isinstance(logits, tuple):
        logits = logits[0]
    m = compute_metrics(labels, np.argmax(logits, axis=-1))
    return {"accuracy": m["accuracy"], "macro_f1": m["macro_f1"],
            "weighted_f1": m["weighted_f1"]}


def train_transformer(train_texts, train_labels, val_texts, val_labels, label_names: List[str],
                      output_dir: Path, model_name: str = "csebuetnlp/banglabert",
                      max_length: int = 128, epochs: int = 4, batch_size: int = 16,
                      lr: float = 2e-5, weight_decay: float = 0.01,
                      use_class_weights: bool = True, use_bangla_normalizer: bool = True,
                      seed: int = 42):
    set_seed(seed)
    output_dir = Path(output_dir)
    normalize = get_text_normalizer(use_bangla_normalizer)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=len(label_names),
        id2label=dict(enumerate(label_names)),
        label2id={l: i for i, l in enumerate(label_names)},
        ignore_mismatched_sizes=True,     # allows starting from a checkpoint with another head
    )

    train_ds = EncodedTextDataset([normalize(t) for t in train_texts], train_labels,
                                  tokenizer, max_length)
    val_ds = EncodedTextDataset([normalize(t) for t in val_texts], val_labels,
                                tokenizer, max_length)

    total_steps = math.ceil(len(train_ds) / batch_size) * epochs
    args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        num_train_epochs=epochs,
        weight_decay=weight_decay,
        warmup_steps=int(0.1 * total_steps),
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        save_total_limit=1,
        fp16=torch.cuda.is_available(),
        report_to="none",
        seed=seed,
    )

    weights = None
    if use_class_weights:
        counts = np.bincount(np.asarray(train_labels), minlength=len(label_names))
        weights = torch.tensor(len(train_labels) / (len(label_names) * np.maximum(counts, 1)),
                               dtype=torch.float32)

    trainer = WeightedTrainer(
        model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=_hf_metrics, class_weights=weights,
    )
    trainer.train()

    best_dir = output_dir / "best"
    trainer.save_model(str(best_dir))
    tokenizer.save_pretrained(str(best_dir))
    (best_dir / "bangla_preprocess.json").write_text(json.dumps({
        "base_model": model_name, "max_length": max_length,
        "use_bangla_normalizer": use_bangla_normalizer, "label_names": label_names,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return trainer.model, tokenizer, normalize


@torch.no_grad()
def predict_transformer(model, tokenizer, texts: Sequence[str], normalize: Callable[[str], str],
                        max_length: int = 128, batch_size: int = 32) -> np.ndarray:
    model.eval()
    device = next(model.parameters()).device
    probs = []
    for i in range(0, len(texts), batch_size):
        batch = [normalize(t) for t in texts[i: i + batch_size]]
        enc = tokenizer(batch, padding=True, truncation=True, max_length=max_length,
                        return_tensors="pt").to(device)
        probs.append(torch.softmax(model(**enc).logits, dim=-1).float().cpu().numpy())
    return np.concatenate(probs)


def load_transformer(model_dir: str, device: Optional[str] = None):
    model_dir = Path(model_dir)
    meta = json.loads((model_dir / "bangla_preprocess.json").read_text(encoding="utf-8"))
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir)).to(device)
    normalize = get_text_normalizer(meta["use_bangla_normalizer"])
    return model, tokenizer, normalize, meta


def run_transformer_experiment(train, val, test, label_names: List[str], out_dir: Path,
                               model_name: str = "csebuetnlp/banglabert", **kwargs) -> List[dict]:
    tag = model_name.split("/")[-1]
    t0 = time.time()
    model, tokenizer, normalize = train_transformer(
        train["text"].tolist(), train["label_id"].tolist(),
        val["text"].tolist(), val["label_id"].tolist(), label_names,
        output_dir=Path(out_dir) / "models" / f"transformer__{tag}",
        model_name=model_name, **kwargs)
    train_time = time.time() - t0

    max_length = kwargs.get("max_length", 128)
    val_pred = predict_transformer(model, tokenizer, val["text"].tolist(), normalize,
                                   max_length).argmax(1)
    test_pred = predict_transformer(model, tokenizer, test["text"].tolist(), normalize,
                                    max_length).argmax(1)
    val_m = compute_metrics(val["label_id"].values, val_pred, label_names)
    test_m = evaluate_and_save(f"transformer__{tag}", test["label_id"].values, test_pred,
                               label_names, out_dir, texts=test["text"].tolist())
    return [{"model": f"transformer__{tag}", "family": "Transformer",
             "train_time_s": train_time, "val_macro_f1": val_m["macro_f1"],
             **{f"test_{k}": v for k, v in test_m.items()}}]
