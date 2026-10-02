import numpy as np
import pytest
import torch

from patchprune import FILL_MODES, METHODS, MarkovEntropyModel, PatchTST, PruningPlan

SEQ_LEN, PRED_LEN, N_VARS = 96, 24, 3


@pytest.fixture(scope="module")
def setup():
    torch.manual_seed(0)
    model = PatchTST(n_vars=N_VARS, seq_len=SEQ_LEN, pred_len=PRED_LEN, d_model=16,
                     n_heads=4, d_ff=32, e_layers=3).eval()
    train = np.random.default_rng(0).standard_normal((600, N_VARS)).astype(np.float32)
    entropy = MarkovEntropyModel.fit(train, SEQ_LEN, max_windows=100)
    x = torch.randn(5, SEQ_LEN, N_VARS)
    return model, entropy, x


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("fill", FILL_MODES)
def test_full_keep_matches_unpruned(setup, method, fill):
    model, entropy, x = setup
    with torch.no_grad():
        ref = model(x)
        out = model(x, PruningPlan(method, keep_ratio=1.0, fill=fill, entropy_model=entropy))
    torch.testing.assert_close(out, ref)


@pytest.mark.parametrize("ratio", [0.8, 0.6, 0.4])
def test_keep_ratio_is_respected(setup, ratio):
    model, entropy, x = setup
    with torch.no_grad():
        _, kept = model(x, PruningPlan("variance", keep_ratio=ratio), return_kept=True)
    expected = round(ratio * model.n_patches)
    assert (kept.sum(1) == expected).all()


@pytest.mark.parametrize("fill", FILL_MODES)
def test_padded_adaptive_path_matches_per_series_computation(setup, fill):
    """The masked (variable-length) path must equal running each series alone."""
    model, entropy, x = setup
    plan = PruningPlan("entropy", top_p=0.7, fill=fill, entropy_model=entropy)
    with torch.no_grad():
        batched, kept = model(x, plan, return_kept=True)
        assert kept.sum(1).unique().numel() > 1, "test needs variable keep counts"
        for i in range(x.size(0)):
            single = model(x[i:i + 1], plan)
            torch.testing.assert_close(batched[i:i + 1], single, rtol=1e-4, atol=1e-5)


def test_recency_keeps_latest_patches(setup):
    model, _, x = setup
    with torch.no_grad():
        _, kept = model(x, PruningPlan("recency", keep_ratio=0.5), return_kept=True)
    k = kept.sum(1)[0].item()
    assert kept[:, -k:].all() and not kept[:, :-k].any()


def test_attention_requires_an_unpruned_layer():
    with pytest.raises(ValueError):
        PruningPlan("attention", keep_ratio=0.5, prune_layer=0)
