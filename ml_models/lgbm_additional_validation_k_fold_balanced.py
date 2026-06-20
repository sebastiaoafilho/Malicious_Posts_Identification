#!/usr/bin/env python3
"""
R3 additional validation for LightGBM + TF-IDF Unigram.

This script reproduces the additional validation experiments added during the
R3 revision of the paper:

1. Stratified 5-fold cross-validation for the unweighted LightGBM model.
2. Stratified 5-fold cross-validation for a class-weighted LightGBM model.
3. PR-AUC and ROC-AUC computation.
4. Threshold sensitivity analysis using out-of-fold probabilities.
5. Optional saving of fold-level and row-level predictions.

Default assumptions:
- Dataset: datasets/dataset_II.csv
- Text column: full_text
- Label column: Relevante
- Positive class: Relevant / Sim / 1
- Text representation: TF-IDF Unigram, L1 normalization, max_features=10000
- Classifier: LightGBM binary classifier

Recommended usage from the repository root:
  python r3_lgbm_additional_validation.py \
    --dataset datasets/dataset_II.csv \
    --text-col full_text \
    --label-col Relevante \
    --output-dir r3_lgbm_outputs

If columns are not provided, the script attempts to infer them.

Outputs:
- r3_lgbm_outputs/cv_summary.csv
- r3_lgbm_outputs/fold_metrics.csv
- r3_lgbm_outputs/threshold_sensitivity.csv
- r3_lgbm_outputs/oof_predictions.csv
- r3_lgbm_outputs/metrics.json
- r3_lgbm_outputs/README.md
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

try:
    from lightgbm import LGBMClassifier
except ImportError as exc:
    raise SystemExit(
        "lightgbm is required. Install it with: pip install lightgbm"
    ) from exc


RANDOM_STATE = 42


@dataclass
class ExperimentConfig:
    dataset: str
    text_col: str
    label_col: str
    positive_class: str
    n_splits: int
    random_state: int
    max_features: int
    ngram_range: Tuple[int, int]
    norm: str
    use_idf: bool
    num_leaves: int
    learning_rate: float
    feature_fraction: float
    n_estimators: int
    objective: str
    metric: str


def set_seed(seed: int = RANDOM_STATE) -> None:
    random.seed(seed)
    np.random.seed(seed)


def infer_text_column(df: pd.DataFrame) -> str:
    """Infer a likely text column from common names or by average string length."""
    candidates = [
        "full_text",
        "processed_text",
        "preprocessed_text",
        "text_cleaned",
        "clean_text",
        "texto_processado",
        "texto_limpo",
        "texto",
        "text",
        "content",
        "question",
        "post",
        "message",
        "body",
        "title_content",
    ]
    lower_map = {c.lower(): c for c in df.columns}
    for candidate in candidates:
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]

    object_cols = [c for c in df.columns if df[c].dtype == "object"]
    if not object_cols:
        raise ValueError("Could not infer text column. Please pass --text-col.")

    lengths = {
        c: df[c].dropna().astype(str).str.len().mean()
        for c in object_cols
    }
    return max(lengths, key=lengths.get)


def infer_label_column(df: pd.DataFrame) -> str:
    """Infer a likely binary label column from common names or binary columns."""
    candidates = [
        "Relevante",
        "relevante",
        "label",
        "labels",
        "class",
        "classe",
        "target",
        "y",
        "relevant",
        "relevance",
        "rotulo",
        "rótulo",
    ]
    lower_map = {c.lower(): c for c in df.columns}
    for candidate in candidates:
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]

    for col in df.columns:
        if df[col].dropna().nunique() == 2:
            return col

    raise ValueError("Could not infer label column. Please pass --label-col.")


def map_labels(series: pd.Series) -> np.ndarray:
    """Map several common label encodings to 0/1, with 1 = Relevant."""
    if pd.api.types.is_numeric_dtype(series):
        values = sorted(pd.Series(series.dropna().unique()).tolist())
        if set(values).issubset({0, 1}):
            return series.astype(int).to_numpy()
        return (series == max(values)).astype(int).to_numpy()

    positive_values = {
        "sim",
        "yes",
        "true",
        "1",
        "relevant",
        "relevante",
        "malicious",
        "potencialmente malicioso",
    }
    negative_values = {
        "não",
        "nao",
        "no",
        "false",
        "0",
        "not relevant",
        "not_relevant",
        "non-relevant",
        "non relevant",
        "irrelevant",
        "nao relevante",
        "não relevante",
    }

    def normalize(value: object) -> int:
        text = str(value).strip().lower()

        if text in positive_values:
            return 1
        if text in negative_values:
            return 0

        if "not relevant" in text or "non-relevant" in text:
            return 0
        if "não relevante" in text or "nao relevante" in text:
            return 0
        if "relevant" in text or "relevante" in text:
            return 1

        raise ValueError(f"Could not map label value: {value!r}")

    return series.map(normalize).astype(int).to_numpy()


def make_vectorizer(max_features: int, ngram_range: Tuple[int, int], norm: str, use_idf: bool) -> TfidfVectorizer:
    return TfidfVectorizer(
        ngram_range=ngram_range,
        norm=norm,
        use_idf=use_idf,
        max_features=max_features,
    )


def make_lgbm(
    scale_pos_weight: Optional[float] = None,
    random_state: int = RANDOM_STATE,
    num_leaves: int = 31,
    learning_rate: float = 0.05,
    feature_fraction: float = 0.9,
    n_estimators: int = 100,
) -> LGBMClassifier:
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "num_leaves": num_leaves,
        "learning_rate": learning_rate,
        "feature_fraction": feature_fraction,
        "n_estimators": n_estimators,
        "random_state": random_state,
        "verbosity": -1,
    }
    if scale_pos_weight is not None:
        params["scale_pos_weight"] = scale_pos_weight

    return LGBMClassifier(**params)


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=[1],
        average=None,
        zero_division=0,
    )
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    return {
        "precision_relevant": float(precision[0]),
        "recall_relevant": float(recall[0]),
        "f1_relevant": float(f1[0]),
        "support_relevant": int(support[0]),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "threshold": float(threshold),
        "predicted_relevant": int(y_pred.sum()),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def summarize_fold_metrics(fold_df: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "precision_relevant",
        "recall_relevant",
        "f1_relevant",
        "pr_auc",
        "roc_auc",
        "accuracy",
    ]
    rows = []
    for model_name, group in fold_df.groupby("model"):
        for metric in metrics:
            rows.append(
                {
                    "model": model_name,
                    "metric": metric,
                    "mean": group[metric].mean(),
                    "std": group[metric].std(ddof=1),
                }
            )
    return pd.DataFrame(rows)


def run_cross_validation(
    texts: np.ndarray,
    y: np.ndarray,
    max_features: int,
    ngram_range: Tuple[int, int],
    norm: str,
    use_idf: bool,
    n_splits: int,
    random_state: int,
    num_leaves: int,
    learning_rate: float,
    feature_fraction: float,
    n_estimators: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float]]:
    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )

    neg_count = int((y == 0).sum())
    pos_count = int((y == 1).sum())
    scale_pos_weight = neg_count / pos_count

    fold_records: List[Dict[str, object]] = []
    oof_records: List[Dict[str, object]] = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(texts, y), start=1):
        x_train, x_test = texts[train_idx], texts[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        vectorizer = make_vectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            norm=norm,
            use_idf=use_idf,
        )
        x_train_vec = vectorizer.fit_transform(x_train)
        x_test_vec = vectorizer.transform(x_test)

        models = {
            "LightGBM": make_lgbm(
                scale_pos_weight=None,
                random_state=random_state,
                num_leaves=num_leaves,
                learning_rate=learning_rate,
                feature_fraction=feature_fraction,
                n_estimators=n_estimators,
            ),
            "LightGBM balanced": make_lgbm(
                scale_pos_weight=scale_pos_weight,
                random_state=random_state,
                num_leaves=num_leaves,
                learning_rate=learning_rate,
                feature_fraction=feature_fraction,
                n_estimators=n_estimators,
            ),
        }

        for model_name, model in models.items():
            model.fit(x_train_vec, y_train)
            y_prob = model.predict_proba(x_test_vec)[:, 1]
            metrics = compute_metrics(y_test, y_prob, threshold=0.5)

            fold_records.append(
                {
                    "model": model_name,
                    "fold": fold,
                    "train_rows": int(len(train_idx)),
                    "test_rows": int(len(test_idx)),
                    "train_relevant": int(y_train.sum()),
                    "test_relevant": int(y_test.sum()),
                    **metrics,
                }
            )

            for idx, prob in zip(test_idx, y_prob):
                oof_records.append(
                    {
                        "row_index": int(idx),
                        "fold": fold,
                        "model": model_name,
                        "y_true": int(y[idx]),
                        "prob_relevant": float(prob),
                    }
                )

    fold_df = pd.DataFrame(fold_records)
    oof_df = pd.DataFrame(oof_records)

    config_values = {
        "scale_pos_weight": float(scale_pos_weight),
        "full_not_relevant": neg_count,
        "full_relevant": pos_count,
    }

    return fold_df, oof_df, config_values


def compute_threshold_sensitivity(oof_df: pd.DataFrame, thresholds: Iterable[float]) -> pd.DataFrame:
    rows = []
    for model_name, group in oof_df.groupby("model"):
        y_true = group["y_true"].to_numpy(dtype=int)
        y_prob = group["prob_relevant"].to_numpy(dtype=float)

        for threshold in thresholds:
            metrics = compute_metrics(y_true, y_prob, threshold=threshold)
            rows.append({"model": model_name, **metrics})

    return pd.DataFrame(rows)


def format_mean_std(summary_df: pd.DataFrame) -> pd.DataFrame:
    """Return a wide, article-friendly summary table."""
    rows = []
    order = [
        ("precision_relevant", "Precision"),
        ("recall_relevant", "Recall"),
        ("f1_relevant", "F1-score"),
        ("pr_auc", "PR-AUC"),
        ("roc_auc", "ROC-AUC"),
        ("accuracy", "Accuracy"),
    ]

    for metric_key, metric_label in order:
        row = {"Metric": metric_label}
        for model_name in ["LightGBM", "LightGBM balanced"]:
            match = summary_df[
                (summary_df["model"] == model_name)
                & (summary_df["metric"] == metric_key)
            ]
            if len(match) == 0:
                row[model_name] = ""
            else:
                mean = match.iloc[0]["mean"]
                std = match.iloc[0]["std"]
                row[model_name] = f"{mean:.3f} ± {std:.3f}"
        rows.append(row)

    return pd.DataFrame(rows)


def write_readme(
    out_dir: Path,
    config: ExperimentConfig,
    class_counts: Dict[str, int],
    scale_pos_weight: float,
    article_summary: pd.DataFrame,
    threshold_df: pd.DataFrame,
) -> None:
    threshold_focus = threshold_df[
        (threshold_df["model"] == "LightGBM")
        & (threshold_df["threshold"].isin([0.3, 0.5, 0.7]))
    ][
        [
            "threshold",
            "precision_relevant",
            "recall_relevant",
            "f1_relevant",
            "predicted_relevant",
        ]
    ]

    readme = f"""# R3 LightGBM additional validation

This folder contains the additional validation experiments for the revised paper.

## Dataset

- Dataset: `{config.dataset}`
- Text column: `{config.text_col}`
- Label column: `{config.label_col}`
- Positive class: `{config.positive_class}`
- Rows: {class_counts["total"]}
- Not Relevant: {class_counts["not_relevant"]}
- Relevant: {class_counts["relevant"]}

## Text representation

- TF-IDF Unigram
- `ngram_range={config.ngram_range}`
- `norm="{config.norm}"`
- `use_idf={config.use_idf}`
- `max_features={config.max_features}`

## Classifier

- LightGBM binary objective
- `num_leaves={config.num_leaves}`
- `learning_rate={config.learning_rate}`
- `feature_fraction={config.feature_fraction}`
- `n_estimators={config.n_estimators}`
- `random_state={config.random_state}`

## Validation

- Stratified {config.n_splits}-fold cross-validation
- Balanced model uses `scale_pos_weight={scale_pos_weight:.6f}`

## Cross-validation summary

{article_summary.to_markdown(index=False)}

## Threshold sensitivity for unweighted LightGBM

{threshold_focus.to_markdown(index=False)}

## Files

- `cv_summary.csv`: mean and standard deviation by metric and model.
- `cv_summary_article_format.csv`: article-friendly table with mean ± std.
- `fold_metrics.csv`: fold-level metrics.
- `threshold_sensitivity.csv`: threshold analysis from 0.1 to 0.9.
- `oof_predictions.csv`: out-of-fold probabilities for both LightGBM variants.
- `metrics.json`: complete machine-readable output.
"""
    (out_dir / "README.md").write_text(readme, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="datasets/dataset_II.csv")
    parser.add_argument("--text-col", default=None)
    parser.add_argument("--label-col", default=None)
    parser.add_argument("--output-dir", default="r3_lgbm_outputs")
    parser.add_argument("--max-features", type=int, default=10000)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=RANDOM_STATE)
    parser.add_argument("--num-leaves", type=int, default=31)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--feature-fraction", type=float, default=0.9)
    parser.add_argument("--n-estimators", type=int, default=100)
    args = parser.parse_args()

    set_seed(args.random_state)

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(dataset_path)
    text_col = args.text_col or infer_text_column(df)
    label_col = args.label_col or infer_label_column(df)

    data = df[[text_col, label_col]].dropna().copy()
    texts = data[text_col].astype(str).to_numpy()
    y = map_labels(data[label_col])

    class_counts = {
        "total": int(len(y)),
        "not_relevant": int((y == 0).sum()),
        "relevant": int((y == 1).sum()),
    }

    config = ExperimentConfig(
        dataset=str(dataset_path),
        text_col=text_col,
        label_col=label_col,
        positive_class="Relevant",
        n_splits=args.n_splits,
        random_state=args.random_state,
        max_features=args.max_features,
        ngram_range=(1, 1),
        norm="l1",
        use_idf=True,
        num_leaves=args.num_leaves,
        learning_rate=args.learning_rate,
        feature_fraction=args.feature_fraction,
        n_estimators=args.n_estimators,
        objective="binary",
        metric="binary_logloss",
    )

    fold_df, oof_df, config_values = run_cross_validation(
        texts=texts,
        y=y,
        max_features=args.max_features,
        ngram_range=(1, 1),
        norm="l1",
        use_idf=True,
        n_splits=args.n_splits,
        random_state=args.random_state,
        num_leaves=args.num_leaves,
        learning_rate=args.learning_rate,
        feature_fraction=args.feature_fraction,
        n_estimators=args.n_estimators,
    )

    summary_df = summarize_fold_metrics(fold_df)
    article_summary_df = format_mean_std(summary_df)

    threshold_values = [round(x, 1) for x in np.arange(0.1, 1.0, 0.1)]
    threshold_df = compute_threshold_sensitivity(oof_df, threshold_values)

    fold_df.to_csv(out_dir / "fold_metrics.csv", index=False)
    oof_df.to_csv(out_dir / "oof_predictions.csv", index=False)
    summary_df.to_csv(out_dir / "cv_summary.csv", index=False)
    article_summary_df.to_csv(out_dir / "cv_summary_article_format.csv", index=False)
    threshold_df.to_csv(out_dir / "threshold_sensitivity.csv", index=False)

    metrics_json = {
        "config": asdict(config),
        "class_counts": class_counts,
        "scale_pos_weight": config_values["scale_pos_weight"],
        "cv_summary": summary_df.to_dict(orient="records"),
        "cv_summary_article_format": article_summary_df.to_dict(orient="records"),
        "threshold_sensitivity": threshold_df.to_dict(orient="records"),
    }
    (out_dir / "metrics.json").write_text(
        json.dumps(metrics_json, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    write_readme(
        out_dir=out_dir,
        config=config,
        class_counts=class_counts,
        scale_pos_weight=config_values["scale_pos_weight"],
        article_summary=article_summary_df,
        threshold_df=threshold_df,
    )

    print("\nCross-validation summary:")
    print(article_summary_df.to_string(index=False))

    print("\nThreshold sensitivity for unweighted LightGBM:")
    focus = threshold_df[
        (threshold_df["model"] == "LightGBM")
        & (threshold_df["threshold"].isin([0.3, 0.5, 0.7]))
    ][
        [
            "threshold",
            "precision_relevant",
            "recall_relevant",
            "f1_relevant",
            "predicted_relevant",
        ]
    ]
    print(focus.to_string(index=False))

    print(f"\nSaved outputs to: {out_dir}")


if __name__ == "__main__":
    main()
