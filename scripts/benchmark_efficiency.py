"""Inference latency, peak activation memory and FLOPs at each pruning level.

All measurements use one fixed batch of test windows, so every configuration
does exactly the same work apart from pruning. Latency includes computing the
pruning scores.

    python scripts/benchmark_efficiency.py --dataset ETTh1 --seed 2021
"""
import argparse
import platform
import subprocess
import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from patchprune import METHODS, MarkovEntropyModel, PruningPlan  # noqa: E402
from patchprune.checkpoint import load_model  # noqa: E402
from patchprune.data import WindowDataset, load_splits  # noqa: E402
from patchprune.profiling import count_flops, measure_latency, peak_memory_bytes  # noqa: E402
from patchprune.utils import get_device, save_json  # noqa: E402

KEEP_RATIOS = [1.0, 0.8, 0.6, 0.4]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True)
    p.add_argument("--seed", type=int, default=2021)
    p.add_argument("--batch_size", type=int, default=32, help="number of windows per forward pass")
    p.add_argument("--fill", default=None, help="defaults to the fill chosen in evaluate_pruning.py")
    p.add_argument("--repeats", type=int, default=50)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--data_dir", default="data")
    p.add_argument("--ckpt_dir", default="checkpoints")
    p.add_argument("--out_dir", default="results")
    p.add_argument("--device", default="auto")
    return p.parse_args()


def cpu_name():
    try:
        return subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return platform.processor() or platform.machine()


def chosen_fill(out_dir, dataset):
    path = Path(out_dir) / f"pruning_{dataset}.csv"
    return pd.read_csv(path).fill.iloc[0] if path.exists() else "skip"


def main():
    args = parse_args()
    torch.set_num_threads(args.threads)
    device = get_device(args.device)
    fill = args.fill or chosen_fill(args.out_dir, args.dataset)

    model, cfg = load_model(args.dataset, args.seed, args.ckpt_dir, device)
    splits = load_splits(args.dataset, args.data_dir, cfg["seq_len"])
    entropy = MarkovEntropyModel.fit(splits.train, cfg["seq_len"], seed=args.seed).to(device)
    test = WindowDataset(splits.test, cfg["seq_len"], cfg["pred_len"])
    x = torch.stack([test[i][0] for i in range(args.batch_size)]).to(device)

    rows = []
    configs = [("none", 1.0)] + [(m, r) for m in METHODS for r in KEEP_RATIOS]
    for method, ratio in configs:
        plan = None if method == "none" else PruningPlan(method, keep_ratio=ratio, fill=fill,
                                                         entropy_model=entropy, seed=args.seed)
        fn = lambda: model(x, plan)  # noqa: E731
        latency, iqr = measure_latency(fn, repeats=args.repeats, device=device)
        rows.append(dict(dataset=args.dataset, method=method, keep_ratio=ratio, fill=fill,
                         latency_ms=latency, latency_iqr_ms=iqr,
                         peak_mem_mb=peak_memory_bytes(fn) / 2**20,
                         gflops=count_flops(fn) / 1e9))
        r = rows[-1]
        print(f"{method:9s} {ratio:.1f} | {latency:7.2f} ms (IQR {iqr:5.2f}) | "
              f"{r['peak_mem_mb']:7.1f} MiB | {r['gflops']:.3f} GFLOPs", flush=True)

    out = Path(args.out_dir)
    pd.DataFrame(rows).to_csv(out / f"efficiency_{args.dataset}.csv", index=False)
    save_json(dict(device=str(device), torch=torch.__version__, threads=args.threads,
                   cpu=cpu_name(), platform=platform.platform(),
                   batch_windows=args.batch_size, series_per_batch=args.batch_size * cfg["n_vars"],
                   repeats=args.repeats),
              out / f"efficiency_{args.dataset}_env.json")


if __name__ == "__main__":
    main()
