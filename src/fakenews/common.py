"""Shared command-line options and result writing for all three trainers."""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from .data import LABELS, load_articles
from .metrics import evaluate, save, summary_line


def base_parser(description: str, default_config: str) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--train", required=True, help="English training file")
    ap.add_argument("--dev", default=None, help="English validation file")
    ap.add_argument("--test", default=None, help="test file (English for 3a, German for 3b)")
    ap.add_argument("--test-lang", choices=["en", "de"], default="en")
    ap.add_argument("--config", default=default_config)
    ap.add_argument("--label-col", default="our rating")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="outputs")
    return ap


def load_all(args):
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    kw = {"label_col": args.label_col}
    train = load_articles(args.train, "en", **kw)
    dev = load_articles(args.dev, "en", **kw) if args.dev else None
    test = load_articles(args.test, args.test_lang, **kw) if args.test else None
    return cfg, train, dev, test


def task_name(args) -> str:
    return "task3b" if args.test_lang == "de" else "task3a"


def report(name: str, split: str, df, pred, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    df = df.drop(columns=["input"]).assign(prediction=[LABELS[i] for i in pred])
    df.to_csv(out_dir / f"{split}_predictions.csv", index=False)
    if "label_id" in df:
        res = evaluate(df["label_id"], pred)
        print(f"{name} [{split}] {summary_line(res)}")
        save(res, out_dir / f"{split}_metrics.json")
