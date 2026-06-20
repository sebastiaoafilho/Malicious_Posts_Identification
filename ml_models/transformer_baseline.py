#!/usr/bin/env python3
"""
Lightweight transformer baseline for R3 revision.

This script fine-tunes a pretrained transformer model on a stratified subset of
DATASET II and reports metrics for the Relevant class, including PR-AUC and ROC-AUC.

Default model:
  neuralmind/bert-base-portuguese-cased (BERTimbau Base)

Recommended usage:
  python r3_transformer_baseline.py \
    --dataset dataset_II.csv \
    --model-name neuralmind/bert-base-portuguese-cased \
    --sample-size 2000 \
    --epochs 2 \
    --batch-size 8 \
    --max-length 128

For a multilingual baseline:
  python r3_transformer_baseline.py \
    --dataset dataset_II.csv \
    --model-name xlm-roberta-base \
    --sample-size 2000 \
    --epochs 2 \
    --batch-size 8 \
    --max-length 128

Outputs:
  r3_transformer_outputs/metrics.json
  r3_transformer_outputs/test_predictions.csv
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)


RANDOM_STATE = 42


def set_seed(seed: int = RANDOM_STATE):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def infer_text_column(df: pd.DataFrame) -> str:
    candidates = [
        "full_text", "processed_text", "preprocessed_text", "text_cleaned",
        "clean_text", "texto_processado", "texto_limpo", "texto", "text",
        "content", "question", "post", "message", "body", "title_content",
    ]
    lower_map = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower_map:
            return lower_map[c.lower()]
    object_cols = [c for c in df.columns if df[c].dtype == "object"]
    if not object_cols:
        raise ValueError("Could not infer text column. Please pass --text-col.")
    lengths = {c: df[c].dropna().astype(str).str.len().mean() for c in object_cols}
    return max(lengths, key=lengths.get)


def infer_label_column(df: pd.DataFrame) -> str:
    candidates = [
        "Relevante", "relevante", "label", "labels", "class", "classe",
        "target", "y", "relevant", "relevance", "rotulo", "rótulo",
    ]
    lower_map = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower_map:
            return lower_map[c.lower()]
    for c in df.columns:
        if df[c].dropna().nunique() == 2:
            return c
    raise ValueError("Could not infer label column. Please pass --label-col.")


def map_labels(s: pd.Series) -> np.ndarray:
    if pd.api.types.is_numeric_dtype(s):
        vals = sorted(pd.Series(s.dropna().unique()).tolist())
        if set(vals).issubset({0, 1}):
            return s.astype(int).to_numpy()
        return (s == max(vals)).astype(int).to_numpy()

    def normalize(v):
        v = str(v).strip().lower()
        negative = [
            "não", "nao", "not relevant", "not_relevant", "non-relevant",
            "non relevant", "irrelevant", "nao relevante", "não relevante",
            "0", "false", "no"
        ]
        positive = [
            "sim", "relevant", "relevante", "1", "true", "yes",
            "malicious", "potencialmente malicioso"
        ]
        if v in negative:
            return 0
        if v in positive:
            return 1
        if "not relevant" in v or "não relevante" in v or "nao relevante" in v:
            return 0
        if "relevant" in v or "relevante" in v:
            return 1
        raise ValueError(f"Could not map label value: {v!r}")

    return s.map(normalize).astype(int).to_numpy()


class TextDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = list(texts)
        self.labels = labels.astype(int)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def evaluate(model, loader, device):
    model.eval()
    all_probs = []
    all_preds = []
    all_labels = []
    total_loss = 0.0
    loss_fn = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch in loader:
            labels = batch["labels"].to(device)
            inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
            outputs = model(**inputs)
            logits = outputs.logits
            loss = loss_fn(logits, labels)
            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = (probs >= 0.5).long()

            total_loss += loss.item() * labels.size(0)
            all_probs.extend(probs.cpu().numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=[1], average=None, zero_division=0
    )
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    metrics = {
        "loss": total_loss / len(y_true),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_relevant": float(precision[0]),
        "recall_relevant": float(recall[0]),
        "f1_relevant": float(f1[0]),
        "support_relevant": int(support[0]),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }
    return metrics, y_prob, y_pred, y_true


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model-name", default="neuralmind/bert-base-portuguese-cased")
    parser.add_argument("--text-col", default=None)
    parser.add_argument("--label-col", default=None)
    parser.add_argument("--sample-size", type=int, default=2000)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--output-dir", default="r3_transformer_outputs")
    parser.add_argument("--use-class-weights", action="store_true")
    args = parser.parse_args()

    set_seed(RANDOM_STATE)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.dataset)
    text_col = args.text_col or infer_text_column(df)
    label_col = args.label_col or infer_label_column(df)

    df = df[[text_col, label_col]].dropna().copy()
    df["label_bin"] = map_labels(df[label_col])

    if args.sample_size and args.sample_size < len(df):
        df_sample, _ = train_test_split(
            df,
            train_size=args.sample_size,
            stratify=df["label_bin"],
            random_state=RANDOM_STATE,
        )
        df = df_sample.reset_index(drop=True)

    train_df, test_df = train_test_split(
        df,
        test_size=args.test_size,
        stratify=df["label_bin"],
        random_state=RANDOM_STATE,
    )

    print(f"Model: {args.model_name}")
    print(f"Text column: {text_col}")
    print(f"Label column: {label_col}")
    print(f"Sample size: {len(df)}")
    print("Sample distribution:", df["label_bin"].value_counts().sort_index().to_dict())
    print("Train distribution:", train_df["label_bin"].value_counts().sort_index().to_dict())
    print("Test distribution:", test_df["label_bin"].value_counts().sort_index().to_dict())

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_dataset = TextDataset(
        train_df[text_col].astype(str).tolist(),
        train_df["label_bin"].to_numpy(),
        tokenizer,
        args.max_length,
    )
    test_dataset = TextDataset(
        test_df[text_col].astype(str).tolist(),
        test_df["label_bin"].to_numpy(),
        tokenizer,
        args.max_length,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    num_training_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * num_training_steps),
        num_training_steps=num_training_steps,
    )

    if args.use_class_weights:
        counts = train_df["label_bin"].value_counts().sort_index()
        weights = torch.tensor(
            [1.0, counts.loc[0] / counts.loc[1]], dtype=torch.float, device=device
        )
        loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
        print(f"Using class weights: {weights.detach().cpu().numpy().tolist()}")
    else:
        loss_fn = torch.nn.CrossEntropyLoss()

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            labels = batch["labels"].to(device)
            inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}

            optimizer.zero_grad()
            outputs = model(**inputs)
            loss = loss_fn(outputs.logits, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item() * labels.size(0)

        avg_train_loss = total_loss / len(train_dataset)
        metrics, _, _, _ = evaluate(model, test_loader, device)
        print(f"Epoch {epoch}: train_loss={avg_train_loss:.4f}, "
              f"test_f1={metrics['f1_relevant']:.4f}, "
              f"test_pr_auc={metrics['pr_auc']:.4f}, "
              f"test_roc_auc={metrics['roc_auc']:.4f}")

    metrics, y_prob, y_pred, y_true = evaluate(model, test_loader, device)

    metrics.update({
        "model_name": args.model_name,
        "sample_size": int(len(df)),
        "train_size": int(len(train_df)),
        "test_size": int(len(test_df)),
        "text_col": text_col,
        "label_col": label_col,
        "max_length": args.max_length,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "random_state": RANDOM_STATE,
        "use_class_weights": bool(args.use_class_weights),
    })

    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False))

    pred_df = test_df.copy()
    pred_df["y_true"] = y_true
    pred_df["y_pred"] = y_pred
    pred_df["prob_relevant"] = y_prob
    pred_df.to_csv(out_dir / "test_predictions.csv", index=False)

    print("\nFinal test metrics:")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"\nSaved outputs to: {out_dir}")


if __name__ == "__main__":
    main()
