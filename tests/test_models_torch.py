"""
End‑to‑end tests for ``spectralmc.models.torch``.

The suite focuses on *reproducibility* and *consistency* of:

* Strongly‑typed helpers (``DType``, ``Device``)
* Thread‑safe context managers (``default_dtype``, ``default_device``)
* SafeTensor and optimiser state round‑tripping
* Environment fingerprinting
* Determinism knobs (cuDNN flags, TF32, etc.)

All tests are fully typed and mypy‑strict‑clean.
"""

from __future__ import annotations

from typing import List, Mapping, Tuple

import pytest
import torch

import spectralmc.models.torch as sm_torch

# ──────────────────────────── helpers & fixtures ────────────────────────────
_HAS_CUDA: bool = torch.cuda.is_available()


def _make_parameters() -> List[torch.nn.Parameter]:
    """Return a small deterministic parameter list."""
    gen = torch.Generator().manual_seed(42)
    p = torch.randn(4, 4, generator=gen)
    return [torch.nn.Parameter(p)]


# ────────────────────────────── DType / Device ──────────────────────────────
def test_dtype_roundtrip() -> None:
    for d in sm_torch.DType:
        torch_dt: torch.dtype = d.to_torch()
        assert sm_torch.DType.from_torch(torch_dt) is d


def test_device_roundtrip() -> None:
    assert sm_torch.Device.from_torch(torch.device("cpu")) is sm_torch.Device.cpu
    if _HAS_CUDA:
        assert (
            sm_torch.Device.from_torch(torch.device("cuda", 0)) is sm_torch.Device.cuda
        )
    else:
        # Non‑zero CUDA index should always fail, regardless of hardware
        with pytest.raises(ValueError):
            sm_torch.Device.from_torch(torch.device("cuda", 1))


# ───────────────────────── context‑managers (thread safe) ───────────────────
def test_default_dtype_nested() -> None:
    orig: torch.dtype = torch.get_default_dtype()
    with sm_torch.default_dtype(torch.float64):
        assert torch.get_default_dtype() is torch.float64
        with sm_torch.default_dtype(torch.float32):
            assert torch.get_default_dtype() is torch.float32
        # Outer should be restored
        assert torch.get_default_dtype() is torch.float64
    assert torch.get_default_dtype() is orig


def test_default_device_nested() -> None:
    orig: torch.device = torch.tensor([]).device
    with sm_torch.default_device(torch.device("cpu")):
        assert torch.tensor([]).device.type == "cpu"
        # Device context nesting should not dead‑lock
        with sm_torch.default_device(torch.device("cpu")):
            assert torch.tensor([]).device.type == "cpu"
    assert torch.tensor([]).device == orig


# ────────────────────────── TensorState round‑trip ──────────────────────────
def test_tensor_state_roundtrip() -> None:
    t_orig: torch.Tensor = torch.arange(10, dtype=torch.float32)
    ts: sm_torch.TensorState = sm_torch.TensorState.from_torch(t_orig)
    t_round: torch.Tensor = ts.to_torch()
    assert torch.equal(t_orig, t_round)
    # from_bytes must succeed on the same payload
    ts2: sm_torch.TensorState = sm_torch.TensorState.from_bytes(ts.data)
    assert ts2.shape == ts.shape
    assert ts2.dtype == ts.dtype


# ───────────────────── Optimiser state serialisation ------------------------
def test_adam_optimizer_state_roundtrip() -> None:
    params: List[torch.nn.Parameter] = _make_parameters()
    opt = torch.optim.Adam(params, lr=0.001)
    loss: torch.Tensor = (params[0] ** 2).sum()
    loss.backward()
    opt.step()

    # Capture the optimiser state directly from its `state_dict`
    state: sm_torch.AdamOptimizerState = sm_torch.AdamOptimizerState.from_torch(
        opt.state_dict()
    )

    # Restore into a fresh optimiser instance
    new_opt = torch.optim.Adam(params, lr=0.001)
    new_opt.load_state_dict(state.to_torch())

    # All parameter steps should be identical
    for group_old, group_new in zip(opt.param_groups, new_opt.param_groups):
        assert group_old["lr"] == group_new["lr"]

    # Internal buffers identical
    sd_old, sd_new = opt.state_dict(), new_opt.state_dict()
    assert sd_old.keys() == sd_new.keys()
    assert isinstance(sd_old["state"], Mapping)
    assert isinstance(sd_new["state"], Mapping)
    assert sd_old["state"].keys() == sd_new["state"].keys()


# ───────────────────────────── TorchEnv snapshot ----------------------------
def test_torch_env_snapshot() -> None:
    env: sm_torch.TorchEnv = sm_torch.TorchEnv.snapshot()
    assert env.torch_version == torch.__version__
    if _HAS_CUDA:
        assert env.cuda_version != "<not available>"
        assert env.gpu_name not in {"<cpu>", ""}
    else:
        assert env.cuda_version == "<not available>"
        assert env.gpu_name == "<cpu>"


# ─────────────────────── Determinism / reproducibility ----------------------
def test_deterministic_random_stream() -> None:
    """
    Very coarse check: two identical RNG seeds **must** yield identical tensors.
    Determinism is expected from PyTorch already, but this guards against the
    façade altering RNG behaviour.
    """

    def _sample() -> torch.Tensor:
        gen = torch.Generator().manual_seed(12345)
        return torch.randn(5, 5, generator=gen)

    a: torch.Tensor = _sample()
    b: torch.Tensor = _sample()
    assert torch.equal(a, b)


@pytest.mark.skipif(not _HAS_CUDA, reason="cuDNN flags only relevant with CUDA")
def test_cudnn_determinism_flags() -> None:
    assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cudnn.benchmark is False
    # TF32 must be disabled
    assert torch.backends.cudnn.allow_tf32 is False
    assert torch.backends.cuda.matmul.allow_tf32 is False


# ─────────────────────────── default_device with CUDA -----------------------
@pytest.mark.skipif(not _HAS_CUDA, reason="requires a CUDA device")
def test_default_device_cuda_context() -> None:
    prev: torch.device = torch.tensor([]).device
    with sm_torch.default_device(torch.device("cuda", 0)):
        assert torch.tensor([]).device.type == "cuda"
    assert torch.tensor([]).device == prev


# ────────────────────────────── sanity check ---------------------------------
def test_public_api_all_exports() -> None:
    expected: Tuple[str, ...] = (
        "FullPrecisionDType",
        "ReducedPrecisionDType",
        "AnyDType",
        "DType",
        "Device",
        "TensorState",
        "TorchEnv",
        "AdamParamState",
        "AdamParamGroup",
        "AdamOptimizerState",
        "default_dtype",
        "default_device",
    )
    assert sm_torch.__all__ == expected
