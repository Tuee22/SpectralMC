# tests/test_models_cpu_gpu_transfer.py
from __future__ import annotations

from types import ModuleType
from typing import List, Union

import importlib
import pytest
import torch

###############################################################################
# Load the module under test                                                  #
###############################################################################

cpu_gpu_transfer: ModuleType = importlib.import_module("spectralmc.models.cpu_gpu_transfer")

# Import the public alias directly so mypy can “see” it.
from spectralmc.models.cpu_gpu_transfer import TensorTree as _TensorTree  # noqa: E402


###############################################################################
# Minimal stub for `spectralmc.models.torch.Device`                           #
###############################################################################


class DummyDevice:
    """Test‑only replacement implementing just `.to_torch()`."""

    def __init__(self, torch_dev: Union[str, torch.device]) -> None:
        self._dev = torch.device(torch_dev)

    def to_torch(self) -> torch.device:  # pragma: no cover
        return self._dev


# Ensure the code under test uses the stub.
setattr(cpu_gpu_transfer, "Device", DummyDevice)


###############################################################################
# Helper – flatten nested TensorTree to a list of tensors                     #
###############################################################################


def _flatten_tensors(tree: _TensorTree) -> List[torch.Tensor]:
    """Pre‑order traversal that returns every `torch.Tensor` leaf in *tree*."""
    result: List[torch.Tensor] = []

    if isinstance(tree, torch.Tensor):
        result.append(tree)

    elif isinstance(tree, list):
        for item in tree:
            result.extend(_flatten_tensors(item))

    elif isinstance(tree, tuple):
        for item in tree:
            result.extend(_flatten_tensors(item))

    elif isinstance(tree, dict):
        for val in tree.values():
            result.extend(_flatten_tensors(val))

    # Scalar leaves yield nothing
    return result


###############################################################################
# Tests that do **not** rely on CUDA                                          #
###############################################################################


def test_already_on_destination_raises() -> None:
    """Moving a tensor/tree to the device it already occupies must fail."""
    tree: _TensorTree = [torch.ones(2, 3), {"foo": 123}]
    with pytest.raises(ValueError):
        cpu_gpu_transfer.move_tensor_tree(tree, dest=DummyDevice("cpu"), pin_memory=False)


def test_unsupported_destination_device() -> None:
    """An unsupported device string triggers a RuntimeError."""
    tensor = torch.zeros(4)
    with pytest.raises(RuntimeError, match="Unsupported destination device"):
        cpu_gpu_transfer.move_tensor_tree(tensor, dest=DummyDevice("xpu"))


def test_cuda_requested_but_not_available(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    If CUDA is reported unavailable and the caller explicitly requests it,
    `move_tensor_tree` must raise before attempting any copy.
    """
    monkeypatch.setattr(cpu_gpu_transfer.torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA destination requested"):
        cpu_gpu_transfer.move_tensor_tree(torch.tensor([1.0]), dest=DummyDevice("cuda:0"))


###############################################################################
# GPU‑dependent tests – these **must** run; absence of CUDA fails the suite   #
###############################################################################


def _assert_cuda_available() -> None:
    if not torch.cuda.is_available():
        pytest.fail("CUDA device required for this test‑suite but none detected.")


def test_cpu_to_cuda_and_back_roundtrip() -> None:
    """
    • Copy an arbitrarily‑nested tree CPU → CUDA and validate equality, structure
      and deep‑copy semantics.
    • Copy back CUDA → CPU (pinned) and perform analogous checks.
    """
    _assert_cuda_available()

    # Build a sample nested structure
    orig_tree: _TensorTree = {
        "a": torch.randn(2, 3),
        "b": [
            torch.arange(5, dtype=torch.int64),
            (torch.zeros(1), 3.1415),
        ],
        "meta": "scalar value",
    }

    # ----------------------------- CPU ➜ CUDA --------------------------------
    to_cuda = cpu_gpu_transfer.move_tensor_tree(orig_tree, dest=DummyDevice("cuda:0"))
    src_flat, dst_flat = _flatten_tensors(orig_tree), _flatten_tensors(to_cuda)

    assert len(src_flat) == len(dst_flat)
    for src_t, dst_t in zip(src_flat, dst_flat):
        assert src_t.device.type == "cpu"
        assert dst_t.device.type == "cuda" and dst_t.device.index == 0
        assert torch.equal(src_t, dst_t.cpu())
        # different storage pointer ⇒ deep copy
        assert src_t.untyped_storage().data_ptr() != dst_t.untyped_storage().data_ptr()

    # ----------------------------- CUDA ➜ CPU --------------------------------
    roundtrip = cpu_gpu_transfer.move_tensor_tree(
        to_cuda, dest=DummyDevice("cpu"), pin_memory=True
    )
    src_flat2, dst_flat2 = _flatten_tensors(to_cuda), _flatten_tensors(roundtrip)

    assert len(src_flat2) == len(dst_flat2)
    for src_t, dst_t in zip(src_flat2, dst_flat2):
        assert src_t.device.type == "cuda"
        assert dst_t.device.type == "cpu"
        assert torch.equal(src_t.cpu(), dst_t)
        assert dst_t.is_pinned()


@pytest.mark.parametrize("pin_memory", [True, False])
def test_gpu_to_cpu_pin_memory_respected(pin_memory: bool) -> None:
    """The `pin_memory` flag must propagate to each CPU allocation."""
    _assert_cuda_available()

    gpu_tensor = torch.randn(8, 8, device="cuda")
    cpu_tensor = cpu_gpu_transfer.move_tensor_tree(
        gpu_tensor, dest=DummyDevice("cpu"), pin_memory=pin_memory
    )
    assert cpu_tensor.device.type == "cpu"
    assert cpu_tensor.is_pinned() == pin_memory
