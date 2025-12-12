# tests/test_models_cpu_gpu_transfer.py
"""Functional-style integration tests for the CPU ↔ GPU tensor copier.

CPU/GPU transfer tests.

These tests intentionally use CPU operations to verify transfer functionality.
"""

from __future__ import annotations

from collections.abc import Mapping

import pytest
import torch

from spectralmc.models import cpu_gpu_transfer
from spectralmc.models.cpu_gpu_transfer import TransferDestination
from spectralmc.models.torch import Device, DType  # façade first
from tests.helpers import expect_success
from spectralmc.result import Failure


###############################################################################
# Utilities
###############################################################################


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

@pytest.fixture(params=[True, False], ids=["pinned_required", "staged_allowed"])
def pinned_required(request: pytest.FixtureRequest) -> bool:
    """Toggle pinned requirement to exercise planner variants."""
    return bool(request.param)

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


def test_cpu_to_cuda_and_back_roundtrip(pinned_required: bool) -> None:
    """Test GPU→CPU transfer and roundtrip (intentional CPU usage)."""
    original: cpu_gpu_transfer.TensorTree = {
        "a": torch.randn(2, 3),
        "b": [torch.arange(5), (torch.zeros(1), 3.14)],
        "meta": "hello",
    }

    to_cuda = expect_success(
        cpu_gpu_transfer.move_tensor_tree(
            original,
            dest=TransferDestination.CUDA,
        )
    )

    src: list[torch.Tensor] = _flatten(original)
    dst: list[torch.Tensor] = _flatten(to_cuda)

    comparisons = [
        (torch.equal(s, d.cpu()), s.device.type, d.device.type) for s, d in zip(src, dst)
    ]
    assert all(
        eq and src_dev == "cpu" and dst_dev == "cuda" for eq, src_dev, dst_dev in comparisons
    ), comparisons

    back = expect_success(
        cpu_gpu_transfer.move_tensor_tree(to_cuda, dest=TransferDestination.CPU_PINNED)
    )

    src2: list[torch.Tensor] = _flatten(to_cuda)
    dst2: list[torch.Tensor] = _flatten(back)

    assert all(torch.equal(s.cpu(), d) and d.is_pinned() for s, d in zip(src2, dst2))



def test_unpinned_host_to_cuda_rejected_when_staging_disabled() -> None:
    """Unpinned host → CUDA should be rejected when staging is disabled."""
    original: cpu_gpu_transfer.TensorTree = {
        "a": torch.randn(2, 2),
    }

    result = cpu_gpu_transfer.move_tensor_tree(
        original, dest=TransferDestination.CUDA, allow_stage=False
    )

    assert isinstance(result, Failure)


@pytest.mark.parametrize(
    "dest,expected_pinned",
    [
        (TransferDestination.CPU, False),
        (TransferDestination.CPU_PINNED, True),
    ],
)
def test_gpu_to_cpu_pin_memory_respected(dest: TransferDestination, expected_pinned: bool) -> None:
    res = expect_success(
        cpu_gpu_transfer.move_tensor_tree(
            torch.randn(4, 4, device="cuda"),
            dest=dest,
        )
    )

    assert isinstance(res, torch.Tensor) and res.is_pinned() == expected_pinned


###############################################################################
# Device/dtype helpers                                                        #
###############################################################################


def test_module_state_device_dtype() -> None:
    dev, dt = expect_success(
        cpu_gpu_transfer.module_state_device_dtype({"w": torch.randn(2, 2), "b": torch.zeros(2)})
    )
    assert (dev, dt) == (Device.cpu, DType.float32)


def test_optimizer_state_device_dtype_after_step() -> None:
    state: Mapping[str, object] = {  # keys are str
        "state": {0: {"exp_avg": torch.ones(1), "exp_avg_sq": torch.ones(1)}},
        "param_groups": [{"lr": 1e-3}],
    }
    assert expect_success(cpu_gpu_transfer.optimizer_state_device_dtype(state)) == (
        Device.cpu,
        DType.float32,
    )


def test_optimizer_state_device_dtype_no_tensors_raises() -> None:
    """Test that empty state returns Failure."""
    result = cpu_gpu_transfer.optimizer_state_device_dtype({"state": {}, "param_groups": []})
    assert isinstance(result, Failure)
