"""Train a PatchTST forecaster once (no pruning during training).

    python scripts/train.py --dataset ETTh1 --seed 2021
"""
import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from patchprune import PatchTST, count_parameters  # noqa: E402
from patchprune.configs import get_config  # noqa: E402
from patchprune.data import load_splits, make_loaders  # noqa: E402
from patchprune.utils import evaluate, get_device, run_name, save_json, set_seed  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True)
    p.add_argument("--seed", type=int, default=2021)
    p.add_argument("--pred_len", type=int)
    p.add_argument("--max_epochs", type=int)
    p.add_argument("--patience", type=int)
    p.add_argument("--batch_size", type=int)
    p.add_argument("--lr", type=float)
    p.add_argument("--train_step", type=int, help="stride between training windows")
    p.add_argument("--data_dir", default="data")
    p.add_argument("--ckpt_dir", default="checkpoints")
    p.add_argument("--device", default="auto")
    p.add_argument("--threads", type=int)
    return p.parse_args()


def adjust_lr(optimizer, epoch, base_lr):
    """PatchTST 'type3' schedule: constant for 3 epochs, then decay by 0.9 per epoch."""
    lr = base_lr if epoch < 3 else base_lr * 0.9 ** (epoch - 2)
    for g in optimizer.param_groups:
        g["lr"] = lr
    return lr


def main():
    args = parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    cfg = get_config(args.dataset, pred_len=args.pred_len, max_epochs=args.max_epochs,
                     patience=args.patience, batch_size=args.batch_size, lr=args.lr,
                     train_step=args.train_step)
    set_seed(args.seed)
    device = get_device(args.device)

    splits = load_splits(cfg["dataset"], args.data_dir, cfg["seq_len"])
    train_dl, val_dl, test_dl = make_loaders(splits, cfg["seq_len"], cfg["pred_len"],
                                             cfg["batch_size"], cfg["train_step"])
    model = PatchTST(**cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])

    name = run_name(cfg, args.seed)
    ckpt = Path(args.ckpt_dir) / f"{name}.pt"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    print(f"{name}: {count_parameters(model):,} parameters, {len(train_dl.dataset):,} training "
          f"windows, {model.n_patches} patches per series, device={device}")

    best, bad, history = float("inf"), 0, []
    for epoch in range(cfg["max_epochs"]):
        lr = adjust_lr(opt, epoch, cfg["lr"])
        model.train()
        t0, total, steps = time.time(), 0.0, 0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            loss = F.mse_loss(model(x), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total, steps = total + loss.item(), steps + 1
        val_mse, val_mae, _ = evaluate(model, val_dl, device)
        history.append(dict(epoch=epoch + 1, lr=lr, train_mse=total / steps,
                            val_mse=val_mse, val_mae=val_mae, seconds=time.time() - t0))
        flag = ""
        if val_mse < best:
            best, bad, flag = val_mse, 0, " *"
            torch.save({"config": cfg, "seed": args.seed, "state_dict": model.state_dict()}, ckpt)
        else:
            bad += 1
        print(f"epoch {epoch + 1:3d} | train {total / steps:.4f} | val mse {val_mse:.4f} "
              f"mae {val_mae:.4f} | {time.time() - t0:5.1f}s{flag}", flush=True)
        if bad >= cfg["patience"]:
            print(f"early stopping after {epoch + 1} epochs")
            break

    model.load_state_dict(torch.load(ckpt, map_location=device)["state_dict"])
    test_mse, test_mae, _ = evaluate(model, test_dl, device)
    print(f"test mse {test_mse:.4f} mae {test_mae:.4f}")
    save_json(dict(run=name, config=cfg, seed=args.seed, best_val_mse=best, test_mse=test_mse,
                   test_mae=test_mae, parameters=count_parameters(model), history=history),
              Path("results") / "training" / f"{name}.json")


if __name__ == "__main__":
    main()
