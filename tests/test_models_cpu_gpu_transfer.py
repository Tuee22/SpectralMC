# tests/test_models_cpu_gpu_transfer.py
"""Functional‑style integration tests for the CPU ↔ GPU tensor copier."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Hashable, List

import pytest
from spectralmc.models.torch import Device, DType  # façade first
import torch
import spectralmc.models.cpu_gpu_transfer as cpu_gpu_transfer

###############################################################################
# Utilities
###############################################################################


def _flatten(tree: cpu_gpu_transfer.TensorTree) -> List[torch.Tensor]:
    match tree:
        case torch.Tensor() as t:
            return [t]
        case list() | tuple() as seq:
            return [x for item in seq for x in _flatten(item)]
        case Mapping() as mp:
            return [x for v in mp.values() for x in _flatten(v)]
        case _:
            return []


# Fail fast if no CUDA – satisfies user requirement
assert torch.cuda.is_available(), "CUDA device required but none detected."

###############################################################################
# CPU‑only tests                                                              #
###############################################################################


def test_already_on_destination_raises() -> None:
    with pytest.raises(ValueError):
        cpu_gpu_transfer.move_tensor_tree([torch.ones(1), 0], dest=Device.cpu)


class BadDest:
    to_torch = staticmethod(lambda: torch.device("xpu"))  # noqa: D401


def test_unsupported_destination_device() -> None:
    with pytest.raises(RuntimeError):
        cpu_gpu_transfer.move_tensor_tree(torch.zeros(1), dest=BadDest())  # type: ignore[arg-type]


def test_cuda_requested_but_not_available(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError):
        cpu_gpu_transfer.move_tensor_tree(torch.zeros(1), dest=Device.cuda)


###############################################################################
# GPU‑dependent tests                                                         #
###############################################################################


def test_cpu_to_cuda_and_back_roundtrip() -> None:
    original: cpu_gpu_transfer.TensorTree = {
        "a": torch.randn(2, 3),
        "b": [torch.arange(5), (torch.zeros(1), 3.14)],
        "meta": "hello",
    }

    to_cuda = cpu_gpu_transfer.move_tensor_tree(original, dest=Device.cuda)

    src: List[torch.Tensor] = _flatten(original)
    dst: List[torch.Tensor] = _flatten(to_cuda)

    assert all(
        torch.equal(s, d.cpu())
        and s.device.type == "cpu"
        and d.device == torch.device("cuda:0")
        and s.untyped_storage().data_ptr() != d.untyped_storage().data_ptr()
        for s, d in zip(src, dst)
    )

    back = cpu_gpu_transfer.move_tensor_tree(to_cuda, dest=Device.cpu, pin_memory=True)

    src2: List[torch.Tensor] = _flatten(to_cuda)
    dst2: List[torch.Tensor] = _flatten(back)

    assert all(torch.equal(s.cpu(), d) and d.is_pinned() for s, d in zip(src2, dst2))


@pytest.mark.parametrize("pin_memory", [True, False])
def test_gpu_to_cpu_pin_memory_respected(pin_memory: bool) -> None:
    res = cpu_gpu_transfer.move_tensor_tree(
        torch.randn(4, 4, device="cuda"),
        dest=Device.cpu,
        pin_memory=pin_memory,
    )
    assert isinstance(res, torch.Tensor) and res.is_pinned() == pin_memory


###############################################################################
# Device/dtype helpers                                                        #
###############################################################################


def test_module_state_device_dtype() -> None:
    dev, dt = cpu_gpu_transfer.module_state_device_dtype(
        {"w": torch.randn(2, 2), "b": torch.zeros(2)}
    )
    assert (dev, dt) == (Device.cpu, DType.float32)


def test_optimizer_state_device_dtype_after_step() -> None:
    state: Mapping[str, object] = {  # keys are str
        "state": {0: {"exp_avg": torch.ones(1), "exp_avg_sq": torch.ones(1)}},
        "param_groups": [{"lr": 1e-3}],
    }
    assert cpu_gpu_transfer.optimizer_state_device_dtype(state) == (
        Device.cpu,
        DType.float32,
    )


def test_optimizer_state_device_dtype_no_tensors_raises() -> None:
    with pytest.raises(RuntimeError):
        cpu_gpu_transfer.optimizer_state_device_dtype({"state": {}, "param_groups": []})
