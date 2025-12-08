# tests/test_models_cpu_gpu_transfer.py
"""Functional-style integration tests for the CPU ↔ GPU tensor copier.

CPU/GPU transfer tests.

These tests intentionally use CPU operations to verify transfer functionality.
All tests marked with @pytest.mark.cpu decorator.
"""

from __future__ import annotations

from collections.abc import Mapping

import pytest
import torch

from spectralmc.models import cpu_gpu_transfer
from spectralmc.models.cpu_gpu_transfer import TransferDestination
from spectralmc.models.torch import Device, DType  # façade first
from spectralmc.result import Failure, Result, Success
from typing import TypeVar


###############################################################################
# Utilities
###############################################################################

T = TypeVar("T")
E = TypeVar("E")


def _expect_success(result: Result[T, E]) -> T:
    match result:
        case Success(value):
            return value
        case Failure(error):
            raise AssertionError(f"Expected Success, got Failure: {error}")


def _flatten(tree: cpu_gpu_transfer.TensorTree) -> list[torch.Tensor]:
    match tree:
        case torch.Tensor() as t:
            return [t]
        case list() | tuple() as seq:
            return [x for item in seq for x in _flatten(item)]
        case Mapping() as mp:
            return [x for v in mp.values() for x in _flatten(v)]
        case _:
            return []


# Fail fast if no CUDA - satisfies user requirement
assert torch.cuda.is_available(), "CUDA device required but none detected."

###############################################################################
# CPU-only tests                                                              #
###############################################################################


def test_already_on_destination_raises() -> None:
    """Test that transferring to same device returns Failure."""
    result = cpu_gpu_transfer.move_tensor_tree([torch.ones(1), 0], dest=TransferDestination.CPU)
    assert isinstance(result, Failure)


def test_cuda_requested_but_not_available(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that CUDA unavailable returns Failure."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    result = cpu_gpu_transfer.move_tensor_tree(torch.zeros(1), dest=TransferDestination.CUDA)
    assert isinstance(result, Failure)


###############################################################################
# GPU-dependent tests                                                         #
###############################################################################


@pytest.mark.cpu  # Intentional CPU/GPU transfer testing
def test_cpu_to_cuda_and_back_roundtrip() -> None:
    """Test GPU→CPU transfer and roundtrip (intentional CPU usage)."""
    original: cpu_gpu_transfer.TensorTree = {
        "a": torch.randn(2, 3),
        "b": [torch.arange(5), (torch.zeros(1), 3.14)],
        "meta": "hello",
    }

    match cpu_gpu_transfer.move_tensor_tree(original, dest=TransferDestination.CUDA):
        case Success(to_cuda):
            pass
        case Failure(error):
            pytest.fail(f"move_tensor_tree failed: {error}")

    src: list[torch.Tensor] = _flatten(original)
    dst: list[torch.Tensor] = _flatten(to_cuda)

    assert all(
        torch.equal(s, d.cpu())
        and s.device.type == "cpu"
        and d.device == torch.device("cuda:0")
        and s.untyped_storage().data_ptr() != d.untyped_storage().data_ptr()
        for s, d in zip(src, dst)
    )

    match cpu_gpu_transfer.move_tensor_tree(to_cuda, dest=TransferDestination.CPU_PINNED):
        case Success(back):
            pass
        case Failure(error):
            pytest.fail(f"move_tensor_tree failed: {error}")

    src2: list[torch.Tensor] = _flatten(to_cuda)
    dst2: list[torch.Tensor] = _flatten(back)

    assert all(torch.equal(s.cpu(), d) and d.is_pinned() for s, d in zip(src2, dst2))


@pytest.mark.parametrize(
    "dest,expected_pinned",
    [
        (TransferDestination.CPU, False),
        (TransferDestination.CPU_PINNED, True),
    ],
)
def test_gpu_to_cpu_pin_memory_respected(dest: TransferDestination, expected_pinned: bool) -> None:
    match cpu_gpu_transfer.move_tensor_tree(
        torch.randn(4, 4, device="cuda"),
        dest=dest,
    ):
        case Success(res):
            pass
        case Failure(error):
            pytest.fail(f"move_tensor_tree failed: {error}")

    assert isinstance(res, torch.Tensor) and res.is_pinned() == expected_pinned


###############################################################################
# Device/dtype helpers                                                        #
###############################################################################


def test_module_state_device_dtype() -> None:
    dev, dt = _expect_success(
        cpu_gpu_transfer.module_state_device_dtype({"w": torch.randn(2, 2), "b": torch.zeros(2)})
    )
    assert (dev, dt) == (Device.cpu, DType.float32)


def test_optimizer_state_device_dtype_after_step() -> None:
    state: Mapping[str, object] = {  # keys are str
        "state": {0: {"exp_avg": torch.ones(1), "exp_avg_sq": torch.ones(1)}},
        "param_groups": [{"lr": 1e-3}],
    }
    assert _expect_success(cpu_gpu_transfer.optimizer_state_device_dtype(state)) == (
        Device.cpu,
        DType.float32,
    )


def test_optimizer_state_device_dtype_no_tensors_raises() -> None:
    """Test that empty state returns Failure."""
    result = cpu_gpu_transfer.optimizer_state_device_dtype({"state": {}, "param_groups": []})
    assert isinstance(result, Failure)
