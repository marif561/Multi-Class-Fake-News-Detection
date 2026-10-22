"""Bi-LSTM checks (skipped when PyTorch is not installed)."""
from pathlib import Path

import pytest
import yaml

torch = pytest.importorskip("torch")

from fakenews.train_bilstm import BiLSTMClassifier, build_vocab, encode, main  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def test_vocab_and_encoding():
    vocab = build_vocab(["a b c", "a b"], max_size=10)
    X = encode(["a z", "b"], vocab, max_len=4)
    assert X.shape == (2, 4) and X[0, 1].item() == 1  # unknown word -> <unk>


def test_forward_backward():
    model = BiLSTMClassifier(20, 8, 6, 10, 0.2, 4)
    x = torch.randint(2, 20, (3, 7))
    x[0, 3:] = 0
    out = model(x)
    assert out.shape == (3, 4)
    out.sum().backward()


def test_end_to_end_small(tmp_path):
    cfg = yaml.safe_load(open(ROOT / "configs/bilstm.yaml"))
    cfg.update(max_len=32, epochs=2, embedding_dim=8, hidden=8, dense=8)
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    s = ROOT / "data" / "sample"
    main(["--train", str(s / "en/train.csv"), "--dev", str(s / "en/dev.csv"),
          "--test", str(s / "de/test.csv"), "--test-lang", "de",
          "--config", str(cfg_path), "--out", str(tmp_path)])
    assert (tmp_path / "bilstm_task3b" / "test_metrics.json").exists()
