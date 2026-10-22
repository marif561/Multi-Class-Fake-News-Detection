"""Bi-LSTM classifier (paper Section 4.2), re-implemented in PyTorch.

Embedding (100-d, trained from scratch) -> Bi-LSTM (128 units) -> Dense 256 + ReLU
-> Dropout 0.2 -> softmax over 4 classes. Adam (lr 1e-4), batch 32, 50 epochs.
The vocabulary is built from the English training data only, which is what makes
Task 3b (German test data) a hard zero-shot transfer.
"""
from __future__ import annotations

import copy
from collections import Counter
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn

from .common import base_parser, load_all, report, task_name
from .data import LABELS
from .metrics import evaluate

PAD, UNK = 0, 1


def build_vocab(texts: List[str], max_size: int) -> Dict[str, int]:
    counts = Counter(t for s in texts for t in s.split())
    vocab = {"<pad>": PAD, "<unk>": UNK}
    for w, _ in counts.most_common(max_size - 2):
        vocab[w] = len(vocab)
    return vocab


def encode(texts: List[str], vocab: Dict[str, int], max_len: int) -> torch.Tensor:
    X = np.zeros((len(texts), max_len), dtype="int64")
    for i, s in enumerate(texts):
        ids = [vocab.get(t, UNK) for t in s.split()[:max_len]]
        X[i, :len(ids)] = ids
    return torch.from_numpy(X)


class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int, hidden: int, dense: int, dropout: float, n_labels: int):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD)
        self.lstm = nn.LSTM(emb_dim, hidden, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(nn.Linear(2 * hidden, dense), nn.ReLU(), nn.Dropout(dropout),
                                  nn.Linear(dense, n_labels))

    def forward(self, x):
        lengths = (x != PAD).sum(1).clamp(min=1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(self.emb(x), lengths, batch_first=True, enforce_sorted=False)
        _, (h, _) = self.lstm(packed)
        return self.head(torch.cat([h[-2], h[-1]], -1))  # final forward + backward states


@torch.no_grad()
def predict(model, X, bs, device):
    model.eval()
    return torch.cat([model(X[i:i + bs].to(device)).argmax(-1).cpu() for i in range(0, len(X), bs)]).numpy()


def main(argv=None):
    args = base_parser(__doc__, "configs/bilstm.yaml").parse_args(argv)
    cfg, train, dev, test = load_all(args)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    vocab = build_vocab(train["input"].tolist(), cfg["vocab_size"])
    enc = lambda df: encode(df["input"].tolist(), vocab, cfg["max_len"])  # noqa: E731
    Xtr, ytr = enc(train), torch.as_tensor(train["label_id"].values)
    model = BiLSTMClassifier(len(vocab), cfg["embedding_dim"], cfg["hidden"], cfg["dense"],
                             cfg["dropout"], len(LABELS)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=float(cfg["learning_rate"]))
    loss_fn = nn.CrossEntropyLoss()
    bs = cfg["batch_size"]

    Xdv = enc(dev) if dev is not None else None
    best, best_state = -1.0, None
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        perm, total = torch.randperm(len(Xtr)), 0.0
        for i in range(0, len(Xtr), bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            loss = loss_fn(model(Xtr[idx].to(device)), ytr[idx].to(device))
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        msg = f"epoch {epoch:>3}  loss {total / len(Xtr):.4f}"
        if Xdv is not None and "label_id" in dev:
            f1 = evaluate(dev["label_id"], predict(model, Xdv, 64, device))["macro_f1"]
            msg += f"  dev macro-F1 {f1:.4f}"
            if f1 > best:
                best, best_state = f1, copy.deepcopy(model.state_dict())
        print(msg)
    if best_state is not None:
        model.load_state_dict(best_state)

    out = Path(args.out) / f"bilstm_{task_name(args)}"
    for split, df in (("dev", dev), ("test", test)):
        if df is not None:
            report("Bi-LSTM", split, df, predict(model, enc(df), 64, device), out)


if __name__ == "__main__":
    main()
