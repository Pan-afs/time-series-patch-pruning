"""Per-dataset hyper-parameters.

ETTh1 uses the PatchTST/42 configuration from the original paper. The Weather
model is narrower than the published one (d_model 64 instead of 128) so that
it trains on a laptop CPU or a free Colab runtime in about an hour; it also sees
every 4th training window (Weather is sampled every 10 minutes, so neighbouring
windows are nearly identical) with a correspondingly higher learning rate.
"""

COMMON = dict(
    seq_len=336,
    pred_len=96,
    patch_len=16,
    stride=8,
    e_layers=3,
    lr=1e-4,
    batch_size=128,
    max_epochs=60,
    patience=10,
    train_step=1,
)

DATASETS = {
    "ETTh1": dict(COMMON, n_vars=7, d_model=16, n_heads=4, d_ff=128,
                  dropout=0.3, head_dropout=0.0),
    "weather": dict(COMMON, n_vars=21, d_model=64, n_heads=8, d_ff=128,
                    dropout=0.2, head_dropout=0.0, lr=3e-4, max_epochs=15, patience=3,
                    train_step=4),
}


def get_config(name: str, **overrides) -> dict:
    if name not in DATASETS:
        raise KeyError(f"unknown dataset {name!r}; choose from {sorted(DATASETS)}")
    cfg = dict(DATASETS[name], dataset=name)
    cfg.update({k: v for k, v in overrides.items() if v is not None})
    return cfg
