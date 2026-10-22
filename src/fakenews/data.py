"""Loading CheckThat! 2022 Task 3 files (columns: public_id, title, text, our rating)."""
from __future__ import annotations

from typing import Dict, List

import pandas as pd

from .preprocess import clean_text

LABELS: List[str] = ["false", "partially false", "true", "other"]
LABEL2ID: Dict[str, int] = {l: i for i, l in enumerate(LABELS)}

_ALIASES = {"partially_false": "partially false", "partially-false": "partially false",
            "partly false": "partially false", "mixture": "partially false"}


def normalise_label(label) -> str:
    key = str(label).strip().lower()
    key = _ALIASES.get(key, key)
    if key not in LABEL2ID:
        raise ValueError(f"Unknown label {label!r}; expected one of {LABELS}")
    return key


def load_articles(path: str, lang: str = "en", label_col: str = "our rating",
                  title_col: str = "title", text_col: str = "text", clean: bool = True,
                  remove_stopwords: bool = True) -> pd.DataFrame:
    """Return title + text joined into ``input`` (cleaned), plus ``label``/``label_id`` if present."""
    df = pd.read_csv(path, sep="\t" if path.endswith(".tsv") else ",")
    title = df[title_col].fillna("").astype(str) if title_col in df else ""
    body = df[text_col].fillna("").astype(str) if text_col in df else ""
    raw = (title + " . " + body) if isinstance(title, pd.Series) else body
    out = pd.DataFrame({"id": df.get("public_id", pd.Series(range(len(df)))), "raw": raw})
    out["input"] = [clean_text(t, lang, remove_stopwords) if clean else t for t in out["raw"]]
    if label_col in df:
        out["label"] = [normalise_label(l) for l in df[label_col]]
        out["label_id"] = out["label"].map(LABEL2ID)
    return out
