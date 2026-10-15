"""Evaluation: accuracy / precision / recall / F1, confusion matrices, error analysis."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)


def compute_metrics(y_true: Sequence[int], y_pred: Sequence[int],
                    label_names: Optional[List[str]] = None) -> Dict[str, float]:
    labels = list(range(len(label_names))) if label_names else None
    metrics = {"accuracy": accuracy_score(y_true, y_pred)}
    for avg in ("macro", "weighted"):
        p, r, f, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=labels, average=avg, zero_division=0)
        metrics.update({f"{avg}_precision": p, f"{avg}_recall": r, f"{avg}_f1": f})
    if label_names:
        _, _, f, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=labels, average=None, zero_division=0)
        metrics.update({f"f1_{name}": score for name, score in zip(label_names, f)})
    return {k: float(v) for k, v in metrics.items()}


def plot_confusion_matrix(y_true, y_pred, label_names: List[str], path: Path,
                          title: str = "") -> None:
    k = len(label_names)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(k)))
    cm_norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)

    fig, ax = plt.subplots(figsize=(1.6 * k + 2.5, 1.4 * k + 2))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(k):
        for j in range(k):
            ax.text(j, i, f"{cm[i, j]}\n{cm_norm[i, j]:.0%}", ha="center", va="center",
                    color="white" if cm_norm[i, j] > 0.5 else "black", fontsize=9)
    ax.set_xticks(range(k), labels=label_names, rotation=30, ha="right")
    ax.set_yticks(range(k), labels=label_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title, fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def evaluate_and_save(name: str, y_true, y_pred, label_names: List[str], out_dir: Path,
                      texts: Optional[Sequence[str]] = None) -> Dict[str, float]:
    """Compute metrics and write report, confusion matrix and misclassified samples."""
    eval_dir = Path(out_dir) / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)

    metrics = compute_metrics(y_true, y_pred, label_names)
    report = classification_report(y_true, y_pred, labels=list(range(len(label_names))),
                                   target_names=label_names, digits=4, zero_division=0)
    (eval_dir / f"{name}_report.txt").write_text(report, encoding="utf-8")
    plot_confusion_matrix(y_true, y_pred, label_names,
                          eval_dir / f"{name}_confusion.png", title=name)

    if texts is not None:
        wrong = y_true != y_pred
        pd.DataFrame({
            "text": np.asarray(texts, dtype=object)[wrong],
            "true": [label_names[i] for i in y_true[wrong]],
            "pred": [label_names[i] for i in y_pred[wrong]],
        }).to_csv(eval_dir / f"{name}_errors.csv", index=False, encoding="utf-8-sig")

    print(f"[eval] {name:<34} acc={metrics['accuracy']:.4f} "
          f"P={metrics['macro_precision']:.4f} R={metrics['macro_recall']:.4f} "
          f"F1={metrics['macro_f1']:.4f}")
    return metrics


def summarize_results(results: List[dict], out_dir: Path) -> pd.DataFrame:
    out_dir = Path(out_dir)
    df = pd.DataFrame(results).sort_values("test_macro_f1", ascending=False)
    df.to_csv(out_dir / "summary.csv", index=False)

    cols = ["model", "family", "val_macro_f1", "test_accuracy", "test_macro_precision",
            "test_macro_recall", "test_macro_f1", "test_weighted_f1", "train_time_s"]
    cols = [c for c in cols if c in df.columns]
    print("\n" + "=" * 100)
    print("RESULTS (sorted by test macro-F1)")
    print("=" * 100)
    print(df[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    fig, ax = plt.subplots(figsize=(9, 0.4 * len(df) + 1.5))
    plot_df = df.iloc[::-1]
    ax.barh(plot_df["model"], plot_df["test_macro_f1"], color="#3b7dd8")
    ax.barh(plot_df["model"], plot_df["test_accuracy"], color="none",
            edgecolor="#e07b39", linewidth=1.5, label="accuracy")
    ax.set_xlabel("Test macro-F1 (bars) / accuracy (outline)")
    ax.set_xlim(0, 1)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_dir / "model_comparison.png", dpi=150)
    plt.close(fig)
    return df


def plot_class_distribution(train, val, test, out_dir: Path) -> None:
    counts = pd.DataFrame({
        "train": train["label"].value_counts(),
        "val": val["label"].value_counts(),
        "test": test["label"].value_counts(),
    }).fillna(0)
    ax = counts.plot(kind="bar", figsize=(7, 4), rot=0)
    ax.set_ylabel("count")
    ax.set_title("Class distribution")
    plt.tight_layout()
    plt.savefig(Path(out_dir) / "class_distribution.png", dpi=150)
    plt.close()
