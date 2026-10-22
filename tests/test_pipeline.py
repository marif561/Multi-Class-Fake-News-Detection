from pathlib import Path

import pytest

from fakenews.data import LABELS, load_articles, normalise_label
from fakenews.metrics import evaluate
from fakenews.preprocess import clean_text
from fakenews.train_pac import main as pac_main

ROOT = Path(__file__).resolve().parents[1]
S = ROOT / "data" / "sample"


def test_clean_text_en_and_de():
    assert clean_text("The CAT, sat on https://x.y the mat!!") == "cat sat mat"
    de = clean_text("Die Stadt eröffnet eine neue Bibliothek", lang="de")
    assert de == "stadt eröffnet neue bibliothek"


def test_load_articles():
    df = load_articles(str(S / "en" / "train.csv"))
    assert len(df) == 8 and set(df["label"]) == set(LABELS)
    assert df["raw"][0].startswith("Scientists") and "cheese" in df["input"][0]


def test_label_aliases():
    assert normalise_label("Partially_False") == "partially false"
    with pytest.raises(ValueError):
        normalise_label("satire")


def test_macro_f1_majority_baseline():
    # Predicting only the majority class gives macro recall 0.25, as Bi-LSTM did on Task 3a
    res = evaluate([0, 0, 1, 2, 3], [0, 0, 0, 0, 0])
    assert res["macro_recall"] == 0.25


@pytest.mark.parametrize("test_file,lang", [("en/test.csv", "en"), ("de/test.csv", "de")])
def test_pac_end_to_end(tmp_path, test_file, lang):
    pac_main(["--train", str(S / "en/train.csv"), "--dev", str(S / "en/dev.csv"),
              "--test", str(S / test_file), "--test-lang", lang,
              "--config", str(ROOT / "configs/pac.yaml"), "--out", str(tmp_path)])
    task = "task3b" if lang == "de" else "task3a"
    assert (tmp_path / f"pac_{task}" / "test_metrics.json").exists()
