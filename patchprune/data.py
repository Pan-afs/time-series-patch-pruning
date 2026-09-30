"""Dataset loading with the standard long-term forecasting splits.

Splits and normalisation follow the protocol used by Informer, Autoformer and
PatchTST so that numbers are comparable with published results:

* ETTh1: 12 / 4 / 4 months for train / val / test.
* Weather: 70% / 10% / 20% chronological split.

Each channel is standardised with statistics from the training split only,
and errors are reported on this standardised scale.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

DATA_URLS = {
    "ETTh1": "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv",
    "weather": "https://huggingface.co/datasets/AutonLab/Timeseries-PILE/resolve/main/forecasting/autoformer/weather.csv",
}


@dataclass
class Splits:
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray
    mean: np.ndarray
    std: np.ndarray


def _borders(name: str, n: int, seq_len: int):
    if name.startswith("ETTh"):
        month = 30 * 24
        b1 = [0, 12 * month - seq_len, 16 * month - seq_len]
        b2 = [12 * month, 16 * month, 20 * month]
    else:
        n_train, n_test = int(n * 0.7), int(n * 0.2)
        n_val = n - n_train - n_test
        b1 = [0, n_train - seq_len, n - n_test - seq_len]
        b2 = [n_train, n_train + n_val, n]
    return b1, b2


def load_splits(name: str, data_dir: str, seq_len: int) -> Splits:
    path = Path(data_dir) / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run `python scripts/download_data.py` first.")
    values = pd.read_csv(path).drop(columns=["date"]).to_numpy(dtype=np.float32)
    b1, b2 = _borders(name, len(values), seq_len)

    train_raw = values[b1[0]:b2[0]]
    mean = train_raw.mean(0)
    std = train_raw.std(0)
    std[std == 0] = 1.0
    scaled = (values - mean) / std
    parts = [scaled[s:e] for s, e in zip(b1, b2)]
    return Splits(*parts, mean=mean, std=std)


class WindowDataset(Dataset):
    """Sliding (lookback, horizon) windows over a multivariate series."""

    def __init__(self, data: np.ndarray, seq_len: int, pred_len: int, step: int = 1):
        self.data = torch.from_numpy(np.ascontiguousarray(data))
        self.seq_len, self.pred_len = seq_len, pred_len
        self.starts = np.arange(0, len(data) - seq_len - pred_len + 1, step)

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, i):
        s = self.starts[i]
        x = self.data[s:s + self.seq_len]
        y = self.data[s + self.seq_len:s + self.seq_len + self.pred_len]
        return x, y


def make_loaders(splits: Splits, seq_len: int, pred_len: int, batch_size: int,
                 train_step: int = 1, num_workers: int = 0):
    train = WindowDataset(splits.train, seq_len, pred_len, step=train_step)
    val = WindowDataset(splits.val, seq_len, pred_len)
    test = WindowDataset(splits.test, seq_len, pred_len)
    kw = dict(batch_size=batch_size, num_workers=num_workers)
    return (DataLoader(train, shuffle=True, drop_last=True, **kw),
            DataLoader(val, shuffle=False, **kw),
            DataLoader(test, shuffle=False, **kw))
