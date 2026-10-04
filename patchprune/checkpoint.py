from pathlib import Path

import torch

from .configs import get_config
from .model import PatchTST
from .utils import run_name


def load_model(dataset: str, seed: int, ckpt_dir: str = "checkpoints", device="cpu",
               pred_len: int = None):
    cfg = get_config(dataset, pred_len=pred_len)
    path = Path(ckpt_dir) / f"{run_name(cfg, seed)}.pt"
    state = torch.load(path, map_location=device)
    model = PatchTST(**state["config"]).to(device).eval()
    model.load_state_dict(state["state_dict"])
    return model, state["config"]
