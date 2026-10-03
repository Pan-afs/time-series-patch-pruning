import json
import random
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def run_name(cfg: dict, seed: int) -> str:
    return f"{cfg['dataset']}_sl{cfg['seq_len']}_pl{cfg['pred_len']}_seed{seed}"


def save_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


@torch.no_grad()
def evaluate(model, loader, device, plan=None):
    """Return (mse, mae, fraction of patches kept) over a loader."""
    model.eval()
    se = ae = n = 0.0
    kept_sum = kept_n = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out, kept = model(x, plan, return_kept=True)
        se += torch.sum((out - y) ** 2).item()
        ae += torch.sum((out - y).abs()).item()
        n += y.numel()
        if kept is not None:
            kept_sum += kept.float().sum().item()
            kept_n += kept.numel()
    frac = kept_sum / kept_n if kept_n else 1.0
    return se / n, ae / n, frac
