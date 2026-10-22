"""Fine-tune RoBERTa (paper Section 4.3) or any Hugging Face encoder.

Paper settings: roberta-base with a softmax classification head, Adam (lr 1e-4),
batch size 32, 20 epochs, dropout 0.1. Transformers see the raw title + text
(no stop-word removal), truncated to --max-length tokens. For Task 3b, a multilingual
model such as xlm-roberta-base can be passed with --model.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from transformers import (AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding,
                          Trainer, TrainingArguments)

from .common import base_parser, load_all, report, task_name
from .data import LABELS
from .metrics import evaluate


def main(argv=None):
    ap = base_parser(__doc__, "configs/roberta.yaml")
    ap.add_argument("--model", default=None, help="override the model in the config")
    args = ap.parse_args(argv)
    cfg, train, dev, test = load_all(args)
    name = args.model or cfg["model"]
    tok = AutoTokenizer.from_pretrained(name)

    def ds(df):
        d = {"text": df["raw"].tolist()}
        if "label_id" in df:
            d["labels"] = df["label_id"].astype(int).tolist()
        return Dataset.from_dict(d).map(
            lambda b: tok(b["text"], truncation=True, max_length=cfg["max_length"]),
            batched=True, remove_columns=["text"])

    model = AutoModelForSequenceClassification.from_pretrained(
        name, num_labels=len(LABELS), id2label=dict(enumerate(LABELS)),
        label2id={l: i for i, l in enumerate(LABELS)}, hidden_dropout_prob=cfg["dropout"])
    tag = f"{name.split('/')[-1]}_{task_name(args)}"
    has_dev = dev is not None
    targs = TrainingArguments(
        output_dir=f"{args.out}/{tag}/checkpoints",
        learning_rate=float(cfg["learning_rate"]), per_device_train_batch_size=cfg["batch_size"],
        per_device_eval_batch_size=cfg["batch_size"], num_train_epochs=cfg["epochs"],
        warmup_ratio=cfg.get("warmup_ratio", 0.0), weight_decay=cfg.get("weight_decay", 0.0),
        eval_strategy="epoch" if has_dev else "no", save_strategy="epoch" if has_dev else "no",
        save_total_limit=1, load_best_model_at_end=has_dev, metric_for_best_model="macro_f1",
        report_to=[], seed=args.seed, fp16=torch.cuda.is_available())
    trainer = Trainer(
        model=model, args=targs, train_dataset=ds(train), eval_dataset=ds(dev) if has_dev else None,
        data_collator=DataCollatorWithPadding(tok),
        compute_metrics=lambda p: {"macro_f1": evaluate(p.label_ids, np.argmax(p.predictions, -1))["macro_f1"]})
    trainer.train()

    out = Path(args.out) / tag
    for split, df in (("dev", dev), ("test", test)):
        if df is not None:
            report(name, split, df, np.argmax(trainer.predict(ds(df)).predictions, -1), out)


if __name__ == "__main__":
    main()
