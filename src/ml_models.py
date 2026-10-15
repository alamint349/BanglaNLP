"""
Classical ML experiments: every (feature set x classifier) combination is
trained as a single sklearn Pipeline (raw text -> cleaner -> features -> clf),
so a saved model can be applied directly to new raw Bangla text.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import List, Sequence

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from .evaluation import compute_metrics, evaluate_and_save
from .features import FEATURE_BUILDERS
from .preprocessing import BanglaTextCleaner

CLASSIFIER_BUILDERS = {
    "logreg": lambda seed: LogisticRegression(C=4.0, max_iter=3000, class_weight="balanced"),
    "linear_svm": lambda seed: LinearSVC(C=0.5, max_iter=5000, class_weight="balanced"),
    "naive_bayes": lambda seed: ComplementNB(alpha=0.3),
    "random_forest": lambda seed: RandomForestClassifier(
        n_estimators=300, class_weight="balanced_subsample", n_jobs=-1, random_state=seed),
}


def build_pipeline(feature: str, classifier: str, min_df: int = 1, seed: int = 42,
                   remove_stopwords: bool = False, stem: bool = False) -> Pipeline:
    return Pipeline([
        ("clean", BanglaTextCleaner(remove_stopwords=remove_stopwords, stem=stem)),
        ("features", FEATURE_BUILDERS[feature](min_df)),
        ("clf", CLASSIFIER_BUILDERS[classifier](seed)),
    ])


def run_ml_experiments(train, val, test, label_names: List[str], out_dir: Path,
                       features: Sequence[str] = tuple(FEATURE_BUILDERS),
                       classifiers: Sequence[str] = tuple(CLASSIFIER_BUILDERS),
                       min_df: int = 1, seed: int = 42, remove_stopwords: bool = False,
                       stem: bool = False) -> List[dict]:
    model_dir = Path(out_dir) / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    results = []

    for feat in features:
        for clf in classifiers:
            name = f"{feat}__{clf}"
            pipe = build_pipeline(feat, clf, min_df, seed, remove_stopwords, stem)

            t0 = time.time()
            pipe.fit(train["text"].tolist(), train["label_id"].values)
            train_time = time.time() - t0

            val_m = compute_metrics(val["label_id"].values,
                                    pipe.predict(val["text"].tolist()), label_names)
            test_m = evaluate_and_save(name, test["label_id"].values,
                                       pipe.predict(test["text"].tolist()),
                                       label_names, out_dir, texts=test["text"].tolist())

            joblib.dump({"pipeline": pipe, "label_names": label_names},
                        model_dir / f"{name}.joblib")
            results.append({
                "model": name, "family": "ML", "train_time_s": train_time,
                "val_macro_f1": val_m["macro_f1"],
                **{f"test_{k}": v for k, v in test_m.items()},
            })
    return results
