"""Build the README figures and summary tables from the CSVs in ./results."""
import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from patchprune import MarkovEntropyModel, PruningPlan  # noqa: E402
from patchprune.checkpoint import load_model  # noqa: E402
from patchprune.data import WindowDataset, load_splits  # noqa: E402

DATASETS = {"ETTh1": "ETTh1", "weather": "Weather"}
STYLE = {  # score methods take the first three categorical slots; baselines stay neutral
    "variance": dict(color="#2a78d6", ls="-", marker="o", label="Variance"),
    "entropy": dict(color="#eb6834", ls="-", marker="s", label="Entropy"),
    "attention": dict(color="#1baf7a", ls="-", marker="^", label="Attention"),
    "random": dict(color="#8a8984", ls="--", marker="x", label="Random"),
    "recency": dict(color="#52514e", ls=":", marker="d", label="Recency (shorter look-back)"),
}
REPORT_RATIOS = [1.0, 0.8, 0.6, 0.4]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "axes.titlesize": 11,
    "axes.titleweight": "semibold", "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.8, "legend.frameon": False, "lines.linewidth": 2,
    "lines.markersize": 5, "figure.dpi": 150, "savefig.bbox": "tight",
})


def load(results, prefix):
    frames = [pd.read_csv(p) for d in DATASETS if (p := results / f"{prefix}_{d}.csv").exists()]
    return pd.concat(frames, ignore_index=True) if frames else None


def default_layer(method):
    return 1 if method == "attention" else 0


def main_rows(df):
    """Rows at each method's default pruning point (input, or after layer 1 for attention)."""
    return df[df.prune_layer == df.method.map(default_layer)]


def fixed_summary(df):
    fixed = df[df.rule == "fixed"]
    return (fixed.groupby(["dataset", "method", "prune_layer", "keep_ratio"])
            .agg(mse=("mse", "mean"), mse_std=("mse", "std"), mae=("mae", "mean"),
                 mae_std=("mae", "std"), mse_unpruned=("mse_unpruned", "mean"),
                 seeds=("seed", "nunique"))
            .reset_index())


def plot_accuracy(summary, out):
    datasets = [d for d in DATASETS if d in set(summary.dataset)]
    fig, axes = plt.subplots(1, len(datasets), figsize=(5.2 * len(datasets), 3.9), squeeze=False)
    for ax, ds in zip(axes[0], datasets):
        sub = summary[summary.dataset == ds]
        for method, st in STYLE.items():
            m = sub[sub.method == method].sort_values("keep_ratio")
            x = 100 * m.keep_ratio
            ax.plot(x, m.mse, color=st["color"], ls=st["ls"], marker=st["marker"], label=st["label"],
                    zorder=3 if st["ls"] == "-" else 2)
            if m.seeds.max() > 1:
                ax.fill_between(x, m.mse - m.mse_std, m.mse + m.mse_std, color=st["color"],
                                alpha=0.12, lw=0)
        base = sub.mse_unpruned.iloc[0]
        ax.axhline(base, color=MUTED, lw=1, zorder=1)
        ax.annotate("unpruned", (20, base), xytext=(0, -4), textcoords="offset points",
                    ha="right", va="top", fontsize=8, color=MUTED)
        seeds = int(sub.seeds.max())
        ax.set_title(f"{DATASETS[ds]} (horizon 96, {seeds} seed{'s' if seeds > 1 else ''})")
        ax.set_xlabel("Patches kept (%)")
        ax.set_xlim(102, 18)
        ax.set_xticks(range(100, 10, -10))
        ax.set_ylabel("Test MSE")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def plot_adaptive(df, out):
    datasets = [d for d in DATASETS if d in set(df.dataset)]
    fig, axes = plt.subplots(1, len(datasets), figsize=(5.2 * len(datasets), 3.6), squeeze=False)
    for ax, ds in zip(axes[0], datasets):
        sub = df[(df.dataset == ds) & df.method.isin(["variance", "entropy", "attention"])]
        g = sub.groupby(["method", "rule", "keep_ratio", "top_p"], dropna=False)[["kept_frac", "mse"]]
        g = g.mean().reset_index()
        for method in ["variance", "entropy", "attention"]:
            st = STYLE[method]
            f = g[(g.method == method) & (g.rule == "fixed")].sort_values("kept_frac")
            a = g[(g.method == method) & (g.rule == "adaptive")].sort_values("kept_frac")
            ax.plot(100 * f.kept_frac, f.mse, color=st["color"], ls="-", lw=1.5, label=f"{st['label']}, fixed")
            ax.plot(100 * a.kept_frac, a.mse, color=st["color"], ls="none", marker="o", ms=7,
                    mfc="white", mew=2, label=f"{st['label']}, adaptive")
        ax.set_title(DATASETS[ds])
        ax.set_xlabel("Patches kept, mean over test set (%)")
        ax.set_ylabel("Test MSE")
        ax.set_xlim(102, 18)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def plot_same_layer(summary_all, out):
    """All scores applied after encoder layer 1, the pruning point attention requires."""
    datasets = [d for d in DATASETS if d in set(summary_all.dataset)]
    fig, axes = plt.subplots(1, len(datasets), figsize=(5.2 * len(datasets), 3.6), squeeze=False)
    for ax, ds in zip(axes[0], datasets):
        sub = summary_all[(summary_all.dataset == ds) & (summary_all.prune_layer == 1)]
        base = summary_all[summary_all.dataset == ds].mse_unpruned.iloc[0]
        for method, st in STYLE.items():
            m = sub[sub.method == method].sort_values("keep_ratio")
            x = np.r_[100 * m.keep_ratio[m.keep_ratio < 1].to_numpy(), 100]
            y = np.r_[m.mse[m.keep_ratio < 1].to_numpy(), base]
            ax.plot(x, y, color=st["color"], ls=st["ls"], marker=st["marker"], label=st["label"])
        ax.axhline(base, color=MUTED, lw=1, zorder=1)
        ax.set_title(f"{DATASETS[ds]}: all scores applied after encoder layer 1")
        ax.set_xlabel("Patches kept (%)")
        ax.set_ylabel("Test MSE")
        ax.set_xlim(102, 18)
        ax.set_xticks(range(100, 10, -10))
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def plot_efficiency(eff, out):
    datasets = [d for d in DATASETS if d in set(eff.dataset)]
    metrics = [("latency_ms", "Latency"), ("peak_mem_mb", "Peak activation memory"),
               ("gflops", "FLOPs")]
    fig, axes = plt.subplots(1, len(metrics), figsize=(12, 3.6))
    ds_style = {"ETTh1": "-", "weather": "--"}
    for ax, (col, title) in zip(axes, metrics):
        for ds in datasets:
            sub = eff[eff.dataset == ds]
            base = sub[sub.method == "none"][col].iloc[0]
            for method in ["variance", "entropy", "attention"]:
                m = sub[sub.method == method].sort_values("keep_ratio")
                ax.plot(100 * m.keep_ratio, 100 * m[col] / base, color=STYLE[method]["color"],
                        ls=ds_style[ds], marker=STYLE[method]["marker"],
                        label=f"{STYLE[method]['label']} ({DATASETS[ds]})")
        ax.plot([100, 40], [100, 40], color=MUTED, lw=1, ls=":", zorder=1)
        ax.set_title(f"{title} (% of unpruned)")
        ax.set_xlabel("Patches kept (%)")
        ax.set_xlim(103, 37)
        ax.set_xticks([100, 80, 60, 40])
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.17))
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


@torch.no_grad()
def plot_example(out, dataset="ETTh1", seed=2021, ratio=0.6, window=1000, channel=-1):
    """Which patches each score keeps on one test window (target channel OT)."""
    model, cfg = load_model(dataset, seed)
    splits = load_splits(dataset, "data", cfg["seq_len"])
    entropy = MarkovEntropyModel.fit(splits.train, cfg["seq_len"], seed=seed)
    x, _ = WindowDataset(splits.test, cfg["seq_len"], cfg["pred_len"])[window]
    x = x[None]
    c = channel % x.size(-1)
    series = x[0, :, c].numpy()
    methods = ["variance", "entropy", "attention"]
    fig, axes = plt.subplots(len(methods), 1, figsize=(10, 1.6 * len(methods) + 0.4), sharex=True)
    for ax, method in zip(axes, methods):
        plan = PruningPlan(method, keep_ratio=ratio, entropy_model=entropy)
        _, kept = model(x, plan, return_kept=True)
        kept = kept.view(1, x.size(-1), -1)[0, c].numpy()
        covered = np.zeros(cfg["seq_len"], dtype=bool)
        for j in np.flatnonzero(kept):
            covered[j * cfg["stride"]:j * cfg["stride"] + cfg["patch_len"]] = True
        for t in np.flatnonzero(~covered):
            ax.axvspan(t - 0.5, t + 0.5, color=GRID, lw=0)
        ax.plot(np.arange(len(series)), series, color=STYLE[method]["color"], lw=1.5)
        ax.set_ylabel(STYLE[method]["label"], rotation=0, ha="right", va="center")
        ax.set_yticks([])
        ax.grid(False)
    axes[0].set_title(f"{DATASETS[dataset]}, channel OT, {int(ratio * 100)}% of patches kept: "
                      "shaded time steps are covered by no kept patch", loc="left")
    axes[-1].set_xlabel("Time step in look-back window")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def write_tables(summary, eff, results):
    lines = []
    for ds in [d for d in DATASETS if d in set(summary.dataset)]:
        sub = summary[(summary.dataset == ds) & summary.keep_ratio.isin(REPORT_RATIOS)]
        seeds = int(sub.seeds.max())
        lines.append(f"### {DATASETS[ds]}: test MSE / MAE"
                     + (f" (mean ± std over {seeds} seeds)" if seeds > 1 else "") + "\n")
        header = "| Method | " + " | ".join(f"{int(r * 100)}% kept" for r in REPORT_RATIOS) + " |"
        lines += [header, "|---" * (len(REPORT_RATIOS) + 1) + "|"]
        for method in STYLE:
            cells = []
            for r in REPORT_RATIOS:
                row = sub[(sub.method == method) & np.isclose(sub.keep_ratio, r)].iloc[0]
                if seeds > 1:
                    cells.append(f"{row.mse:.3f} ± {row.mse_std:.3f} / {row.mae:.3f}")
                else:
                    cells.append(f"{row.mse:.3f} / {row.mae:.3f}")
            lines.append(f"| {STYLE[method]['label']} | " + " | ".join(cells) + " |")
        lines.append("")
        if eff is not None and ds in set(eff.dataset):
            e = eff[eff.dataset == ds]
            base = e[e.method == "none"].iloc[0]
            lines.append(f"### {DATASETS[ds]}: inference cost per batch\n")
            lines += ["| Method | Kept | Latency (ms) | Peak memory (MiB) | GFLOPs |",
                      "|---|---|---|---|---|",
                      f"| Unpruned | 100% | {base.latency_ms:.1f} | {base.peak_mem_mb:.1f} | {base.gflops:.3f} |"]
            for method in ["variance", "entropy", "attention"]:
                for r in REPORT_RATIOS[1:]:
                    row = e[(e.method == method) & np.isclose(e.keep_ratio, r)].iloc[0]
                    lines.append(f"| {STYLE[method]['label']} | {int(r * 100)}% | {row.latency_ms:.1f} | "
                                 f"{row.peak_mem_mb:.1f} | {row.gflops:.3f} |")
            lines.append("")
    (results / "summary.md").write_text("\n".join(lines))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", default="results")
    p.add_argument("--figures", default="figures")
    args = p.parse_args()
    results, figures = Path(args.results), Path(args.figures)
    figures.mkdir(exist_ok=True)

    df = load(results, "pruning")
    eff = load(results, "efficiency")
    summary_all = fixed_summary(df)
    summary_all.to_csv(results / "summary_fixed.csv", index=False)
    summary = main_rows(summary_all)
    plot_accuracy(summary, figures / "accuracy_vs_patches_kept.png")
    plot_adaptive(main_rows(df), figures / "adaptive_vs_fixed.png")
    plot_same_layer(summary_all, figures / "same_pruning_point.png")
    if eff is not None:
        plot_efficiency(eff, figures / "efficiency.png")
    plot_example(figures / "example_selection.png")
    write_tables(summary, eff, results)
    print((results / "summary.md").read_text())


if __name__ == "__main__":
    main()
