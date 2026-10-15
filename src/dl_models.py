"""
Deep-learning classifiers for Bangla text (PyTorch).

  bilstm - 2-direction LSTM + additive attention pooling
  cnn    - TextCNN (Kim, 2014) with multiple kernel widths

Embeddings are trained from scratch, or initialised from pretrained fastText
Bangla vectors (cc.bn.300.vec from fasttext.cc) via --embeddings.
Training uses class-weighted loss, gradient clipping, LR scheduling and early
stopping on validation macro-F1.
"""
from __future__ import annotations

import copy
import random
import time
from collections import Counter
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader, Dataset

from .evaluation import compute_metrics, evaluate_and_save
from .preprocessing import BanglaPreprocessor, PreprocessConfig, normalize_unicode

PAD_IDX, UNK_IDX = 0, 1


@dataclass
class DLConfig:
    model_type: str = "bilstm"          # "bilstm" | "cnn"
    embed_dim: int = 300
    hidden_dim: int = 128
    num_layers: int = 1
    num_filters: int = 128
    kernel_sizes: Tuple[int, ...] = (2, 3, 4, 5)
    dropout: float = 0.4
    max_len: int = 128
    min_freq: int = 1
    max_vocab: int = 50_000
    batch_size: int = 64
    epochs: int = 15
    lr: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 3
    embeddings_path: Optional[str] = None
    freeze_embeddings: bool = False
    seed: int = 42


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------------- #
# Vocabulary and data loading
# --------------------------------------------------------------------------- #
class Vocab:
    def __init__(self, itos: Sequence[str]):
        self.itos = list(itos)
        self.stoi = {t: i for i, t in enumerate(self.itos)}

    @classmethod
    def build(cls, token_lists: Sequence[Sequence[str]], min_freq: int = 1,
              max_size: int = 50_000) -> "Vocab":
        counter = Counter(t for toks in token_lists for t in toks)
        words = [w for w, c in counter.most_common() if c >= min_freq][: max_size - 2]
        return cls(["<pad>", "<unk>"] + words)

    def encode(self, tokens: Sequence[str], max_len: int) -> List[int]:
        ids = [self.stoi.get(t, UNK_IDX) for t in tokens[:max_len]]
        return ids or [UNK_IDX]

    def __len__(self) -> int:
        return len(self.itos)


class EncodedDataset(Dataset):
    def __init__(self, sequences: List[List[int]], labels: Optional[Sequence[int]] = None):
        self.sequences = sequences
        self.labels = list(labels) if labels is not None else [-1] * len(sequences)

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, i):
        return self.sequences[i], self.labels[i]


def collate_batch(batch, min_len: int = 1):
    seqs, labels = zip(*batch)
    lengths = torch.tensor([len(s) for s in seqs], dtype=torch.long)
    ids = torch.full((len(seqs), max(int(lengths.max()), min_len)), PAD_IDX, dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, : len(s)] = torch.tensor(s, dtype=torch.long)
    return ids, lengths, torch.tensor(labels, dtype=torch.long)


def make_loader(vocab: Vocab, token_lists, labels, cfg: DLConfig, shuffle: bool) -> DataLoader:
    seqs = [vocab.encode(t, cfg.max_len) for t in token_lists]
    min_len = max(cfg.kernel_sizes) if cfg.model_type == "cnn" else 1
    return DataLoader(EncodedDataset(seqs, labels), batch_size=cfg.batch_size,
                      shuffle=shuffle, collate_fn=partial(collate_batch, min_len=min_len))


def load_pretrained_embeddings(path: str, vocab: Vocab, dim: int) -> torch.Tensor:
    """Load a word2vec/fastText .vec text file for words in the vocabulary."""
    matrix = np.random.normal(0, 0.1, (len(vocab), dim)).astype(np.float32)
    matrix[PAD_IDX] = 0.0
    found = 0
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line_no, line in enumerate(f):
            parts = line.rstrip().split(" ")
            if line_no == 0 and len(parts) == 2:          # header: "<n_words> <dim>"
                continue
            if len(parts) != dim + 1:
                continue
            idx = vocab.stoi.get(normalize_unicode(parts[0]))
            if idx is not None:
                matrix[idx] = np.asarray(parts[1:], dtype=np.float32)
                found += 1
    print(f"[dl] pretrained embeddings cover {found}/{len(vocab)} "
          f"({found / len(vocab):.1%}) vocabulary words")
    return torch.from_numpy(matrix)


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
class BiLSTMAttention(nn.Module):
    def __init__(self, vocab_size: int, num_classes: int, cfg: DLConfig,
                 pretrained: Optional[torch.Tensor] = None):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, cfg.embed_dim, padding_idx=PAD_IDX)
        if pretrained is not None:
            self.embedding.weight.data.copy_(pretrained)
            self.embedding.weight.requires_grad = not cfg.freeze_embeddings
        self.lstm = nn.LSTM(cfg.embed_dim, cfg.hidden_dim, num_layers=cfg.num_layers,
                            batch_first=True, bidirectional=True,
                            dropout=cfg.dropout if cfg.num_layers > 1 else 0.0)
        self.attention = nn.Linear(2 * cfg.hidden_dim, 1)
        self.dropout = nn.Dropout(cfg.dropout)
        self.fc = nn.Linear(2 * cfg.hidden_dim, num_classes)

    def forward(self, ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        mask = ids != PAD_IDX
        emb = self.dropout(self.embedding(ids))
        packed = pack_padded_sequence(emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.lstm(packed)
        out, _ = pad_packed_sequence(out, batch_first=True, total_length=ids.size(1))
        scores = self.attention(out).squeeze(-1).masked_fill(~mask, -1e9)
        weights = torch.softmax(scores, dim=-1)                       # (B, T)
        context = torch.bmm(weights.unsqueeze(1), out).squeeze(1)      # (B, 2H)
        return self.fc(self.dropout(context))


class TextCNN(nn.Module):
    def __init__(self, vocab_size: int, num_classes: int, cfg: DLConfig,
                 pretrained: Optional[torch.Tensor] = None):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, cfg.embed_dim, padding_idx=PAD_IDX)
        if pretrained is not None:
            self.embedding.weight.data.copy_(pretrained)
            self.embedding.weight.requires_grad = not cfg.freeze_embeddings
        self.convs = nn.ModuleList(
            [nn.Conv1d(cfg.embed_dim, cfg.num_filters, k) for k in cfg.kernel_sizes])
        self.dropout = nn.Dropout(cfg.dropout)
        self.fc = nn.Linear(cfg.num_filters * len(cfg.kernel_sizes), num_classes)

    def forward(self, ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        emb = self.dropout(self.embedding(ids)).transpose(1, 2)        # (B, E, T)
        pooled = [F.relu(conv(emb)).max(dim=2).values for conv in self.convs]
        return self.fc(self.dropout(torch.cat(pooled, dim=1)))


MODEL_CLASSES = {"bilstm": BiLSTMAttention, "cnn": TextCNN}


def build_model(cfg: DLConfig, vocab_size: int, num_classes: int,
                pretrained: Optional[torch.Tensor] = None) -> nn.Module:
    return MODEL_CLASSES[cfg.model_type](vocab_size, num_classes, cfg, pretrained)


# --------------------------------------------------------------------------- #
# Training / inference
# --------------------------------------------------------------------------- #
def class_weights(labels: Sequence[int], num_classes: int) -> torch.Tensor:
    counts = np.bincount(np.asarray(labels), minlength=num_classes).astype(np.float32)
    weights = len(labels) / (num_classes * np.maximum(counts, 1.0))
    return torch.tensor(weights, dtype=torch.float32)


@torch.no_grad()
def predict_proba_loader(model: nn.Module, loader: DataLoader, device) -> np.ndarray:
    model.eval()
    probs = []
    for ids, lengths, _ in loader:
        logits = model(ids.to(device), lengths)
        probs.append(torch.softmax(logits, dim=-1).cpu().numpy())
    return np.concatenate(probs)


def train_dl_model(train_tokens, train_labels, val_tokens, val_labels, num_classes: int,
                   cfg: DLConfig, device: Optional[str] = None):
    set_seed(cfg.seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    vocab = Vocab.build(train_tokens, cfg.min_freq, cfg.max_vocab)
    pretrained = (load_pretrained_embeddings(cfg.embeddings_path, vocab, cfg.embed_dim)
                  if cfg.embeddings_path else None)
    model = build_model(cfg, len(vocab), num_classes, pretrained).to(device)

    train_loader = make_loader(vocab, train_tokens, train_labels, cfg, shuffle=True)
    val_loader = make_loader(vocab, val_tokens, val_labels, cfg, shuffle=False)

    criterion = nn.CrossEntropyLoss(weight=class_weights(train_labels, num_classes).to(device))
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max",
                                                           factor=0.5, patience=1)

    print(f"[dl] {cfg.model_type}: vocab={len(vocab)} params="
          f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,} device={device}")
    best_f1, best_state, bad_epochs, history = -1.0, None, 0, []

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        total_loss, n = 0.0, 0
        for ids, lengths, labels in train_loader:
            ids, labels = ids.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(ids, lengths), labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += loss.item() * labels.size(0)
            n += labels.size(0)

        val_pred = predict_proba_loader(model, val_loader, device).argmax(1)
        val_f1 = f1_score(val_labels, val_pred, average="macro", zero_division=0)
        val_acc = float(np.mean(val_pred == np.asarray(val_labels)))
        scheduler.step(val_f1)
        history.append({"epoch": epoch, "train_loss": total_loss / n,
                        "val_acc": val_acc, "val_macro_f1": val_f1})
        print(f"[dl] {cfg.model_type} epoch {epoch:02d} | loss {total_loss / n:.4f} | "
              f"val_acc {val_acc:.4f} | val_macro_f1 {val_f1:.4f}")

        if val_f1 > best_f1 + 1e-4:
            best_f1, bad_epochs = val_f1, 0
            best_state = copy.deepcopy({k: v.cpu() for k, v in model.state_dict().items()})
        else:
            bad_epochs += 1
            if bad_epochs >= cfg.patience:
                print(f"[dl] early stopping at epoch {epoch} (best val macro-F1 {best_f1:.4f})")
                break

    model.load_state_dict(best_state)
    return model, vocab, history


def predict_dl(model, vocab: Vocab, token_lists, cfg: DLConfig, device=None) -> np.ndarray:
    device = device or next(model.parameters()).device
    loader = make_loader(vocab, token_lists, None, cfg, shuffle=False)
    return predict_proba_loader(model, loader, device)


def save_dl_model(path: Path, model, vocab: Vocab, cfg: DLConfig, label_names, preprocess: dict):
    torch.save({"state_dict": model.state_dict(), "itos": vocab.itos, "config": asdict(cfg),
                "label_names": list(label_names), "preprocess": preprocess}, path)


def load_dl_model(path, device: str = "cpu"):
    ckpt = torch.load(path, map_location=device)
    cfg_dict = dict(ckpt["config"])
    cfg_dict["kernel_sizes"] = tuple(cfg_dict["kernel_sizes"])
    cfg = DLConfig(**cfg_dict)
    vocab = Vocab(ckpt["itos"])
    model = build_model(cfg, len(vocab), len(ckpt["label_names"]))
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model, vocab, cfg, ckpt["label_names"], ckpt["preprocess"]


def run_dl_experiments(train, val, test, label_names: List[str], out_dir: Path,
                       model_types: Sequence[str] = ("bilstm", "cnn"),
                       preprocess_cfg: Optional[PreprocessConfig] = None, **cfg_overrides):
    preprocess_cfg = preprocess_cfg or PreprocessConfig()
    pp = BanglaPreprocessor(preprocess_cfg)
    tr_tok, va_tok, te_tok = (pp.batch_tokens(df["text"]) for df in (train, val, test))
    model_dir = Path(out_dir) / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for mtype in model_types:
        cfg = DLConfig(model_type=mtype, **cfg_overrides)
        t0 = time.time()
        model, vocab, _ = train_dl_model(tr_tok, train["label_id"].tolist(), va_tok,
                                         val["label_id"].tolist(), len(label_names), cfg)
        train_time = time.time() - t0

        name = f"dl__{mtype}" + ("_fasttext" if cfg.embeddings_path else "")
        val_m = compute_metrics(val["label_id"].values,
                                predict_dl(model, vocab, va_tok, cfg).argmax(1), label_names)
        test_m = evaluate_and_save(name, test["label_id"].values,
                                   predict_dl(model, vocab, te_tok, cfg).argmax(1),
                                   label_names, out_dir, texts=test["text"].tolist())
        save_dl_model(model_dir / f"{name}.pt", model, vocab, cfg, label_names,
                      preprocess_cfg.to_dict())
        results.append({"model": name, "family": "DL", "train_time_s": train_time,
                        "val_macro_f1": val_m["macro_f1"],
                        **{f"test_{k}": v for k, v in test_m.items()}})
    return results
