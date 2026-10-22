"""TF-IDF + Passive Aggressive Classifier (paper Section 4.1)."""
from __future__ import annotations

from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.pipeline import make_pipeline

from .common import base_parser, load_all, report, task_name


def build(cfg: dict):
    return make_pipeline(
        TfidfVectorizer(max_df=cfg["max_df"]),
        PassiveAggressiveClassifier(C=cfg["C"], max_iter=cfg["max_iter"], random_state=cfg["random_state"]))


def main(argv=None):
    args = base_parser(__doc__, "configs/pac.yaml").parse_args(argv)
    cfg, train, dev, test = load_all(args)
    clf = build(cfg).fit(train["input"], train["label_id"])
    out = Path(args.out) / f"pac_{task_name(args)}"
    for split, df in (("dev", dev), ("test", test)):
        if df is not None:
            report("PAC", split, df, clf.predict(df["input"]), out)


if __name__ == "__main__":
    main()
