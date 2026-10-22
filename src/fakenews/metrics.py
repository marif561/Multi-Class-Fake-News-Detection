"""Accuracy and macro-averaged precision / recall / F1 (the official CheckThat! metric)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Sequence

from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support

from .data import LABELS


def evaluate(y_true: Sequence[int], y_pred: Sequence[int]) -> Dict:
    ids = list(range(len(LABELS)))
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=ids, average="macro", zero_division=0)
    return {"accuracy": accuracy_score(y_true, y_pred), "macro_precision": p, "macro_recall": r,
            "macro_f1": f,
            "per_class": classification_report(y_true, y_pred, labels=ids, target_names=LABELS,
                                               output_dict=True, zero_division=0)}


def summary_line(res: Dict) -> str:
    return (f"acc {res['accuracy']:.4f} · macro P {res['macro_precision']:.4f} · "
            f"R {res['macro_recall']:.4f} · F1 {res['macro_f1']:.4f}")


def save(res: Dict, path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(res, indent=2), encoding="utf-8")
