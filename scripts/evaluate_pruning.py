"""Forecasting error of a trained PatchTST under inference-time patch pruning.

1. Choose how dropped tokens are represented before the head ("fill") on the
   validation split, once per dataset.
2. On the test split, sweep every scoring method over fixed keep ratios and
   the score-based methods over adaptive top-p thresholds.
3. Ablation: prune every method after the first encoder layer, the same point
   the attention score needs, to separate the score from the pruning point.

    python scripts/evaluate_pruning.py --dataset ETTh1 --seeds 2021 2022 2023
"""
import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from patchprune import FILL_MODES, METHODS, MarkovEntropyModel, PruningPlan  # noqa: E402
from patchprune.checkpoint import load_model  # noqa: E402
from patchprune.data import WindowDataset, load_splits  # noqa: E402
from patchprune.utils import evaluate, get_device  # noqa: E402

KEEP_RATIOS = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]
TOP_P = [0.5, 0.7, 0.8, 0.9, 0.95, 0.99]
ADAPTIVE_METHODS = ["variance", "entropy", "attention"]
FILL_SEARCH_RATIOS = [0.8, 0.6, 0.4]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=[2021])
    p.add_argument("--fill", choices=FILL_MODES, help="skip the validation search and use this fill")
    p.add_argument("--val_step", type=int, default=1, help="stride over validation windows")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--data_dir", default="data")
    p.add_argument("--ckpt_dir", default="checkpoints")
    p.add_argument("--out_dir", default="results")
    p.add_argument("--device", default="auto")
    return p.parse_args()


def main():
    args = parse_args()
    device = get_device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows, fill_rows = [], []
    for seed in args.seeds:
        model, cfg = load_model(args.dataset, seed, args.ckpt_dir, device)
        splits = load_splits(args.dataset, args.data_dir, cfg["seq_len"])
        entropy = MarkovEntropyModel.fit(splits.train, cfg["seq_len"], seed=seed).to(device)
        loader = lambda data, step=1: DataLoader(  # noqa: E731
            WindowDataset(data, cfg["seq_len"], cfg["pred_len"], step=step),
            batch_size=args.batch_size, shuffle=False)
        val_dl, test_dl = loader(splits.val, args.val_step), loader(splits.test)

        def run(dl, **plan_kw):
            plan = PruningPlan(entropy_model=entropy, seed=seed, **plan_kw)
            return evaluate(model, dl, device, plan)

        # --- 1. fill mode, chosen on validation ---------------------------------
        if args.fill is None:
            for method in METHODS:
                for fill in FILL_MODES:
                    for ratio in FILL_SEARCH_RATIOS:
                        mse, mae, _ = run(val_dl, method=method, keep_ratio=ratio, fill=fill)
                        fill_rows.append(dict(dataset=args.dataset, seed=seed, method=method,
                                              fill=fill, keep_ratio=ratio, val_mse=mse, val_mae=mae))
            fills = pd.DataFrame(fill_rows)
            fill = fills[fills.seed == seed].groupby("fill").val_mse.mean().idxmin()
        else:
            fill = args.fill
        print(f"[{args.dataset} seed {seed}] fill mode: {fill}", flush=True)

        # --- 2. test-set sweep ------------------------------------------------------
        base_mse, base_mae, _ = evaluate(model, test_dl, device)
        print(f"  unpruned test mse {base_mse:.4f} mae {base_mae:.4f}", flush=True)
        configs = [dict(method=m, keep_ratio=r) for m in METHODS for r in KEEP_RATIOS]
        configs += [dict(method=m, top_p=p) for m in ADAPTIVE_METHODS for p in TOP_P]
        configs += [dict(method=m, keep_ratio=r, prune_layer=1)
                    for m in METHODS if m != "attention" for r in KEEP_RATIOS[1:]]
        for kw in configs:
            t0 = time.time()
            plan = PruningPlan(entropy_model=entropy, seed=seed, fill=fill, **kw)
            mse, mae, kept = evaluate(model, test_dl, device, plan)
            rows.append(dict(dataset=args.dataset, seed=seed, method=kw["method"],
                             rule="adaptive" if "top_p" in kw else "fixed",
                             keep_ratio=kw.get("keep_ratio"), top_p=kw.get("top_p"),
                             prune_layer=plan.prune_layer, fill=fill, kept_frac=kept,
                             mse=mse, mae=mae, mse_unpruned=base_mse, mae_unpruned=base_mae))
            print(f"  {plan.label:55s} kept {kept:.2f} mse {mse:.4f} mae {mae:.4f} "
                  f"({time.time() - t0:.1f}s)", flush=True)

    pd.DataFrame(rows).to_csv(out_dir / f"pruning_{args.dataset}.csv", index=False)
    if fill_rows:
        pd.DataFrame(fill_rows).to_csv(out_dir / f"fill_search_{args.dataset}.csv", index=False)


if __name__ == "__main__":
    main()
