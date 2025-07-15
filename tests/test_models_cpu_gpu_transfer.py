# tests/test_models_cpu_gpu_transfer.py
"""Integration tests for ``spectralmc.models.cpu_gpu_transfer``.

This file verifies correct behaviour of the public helper
:pyfunc:`spectralmc.models.cpu_gpu_transfer.move_tensor_tree` when copying
arbitrarily nested *TensorTree* structures between CPU and CUDA memory.

Why we import via ``importlib.import_module`` instead of a regular import
=======================================================================
The tests monkey-patch the symbol ``Device`` inside the target module with a
light-weight stub class (`DummyDevice`) that does **not** inherit from the real
``spectralmc.models.torch.Device`` type.  That is required to trigger specific
edge-cases (e.g. passing an object that satisfies the runtime API but violates
its static type contract).

Under ``mypy --strict`` the following problems arise with a normal import such
as::

    import spectralmc.models.cpu_gpu_transfer as cpu_gpu_transfer

* The call ``cpu_gpu_transfer.move_tensor_tree(..., dest=DummyDevice("cpu"))``
  raises *arg-type* because *dest* is not a ``Device``.
* The statement ``cpu_gpu_transfer.Device = DummyDevice`` raises *attr-defined*
  because mypy does not track dynamic attribute mutation.

By retrieving the module dynamically::

    import importlib, types
    cpu_gpu_transfer: types.ModuleType = importlib.import_module(
        "spectralmc.models.cpu_gpu_transfer"
    )

and annotating it with :class:`types.ModuleType`, we deliberately tell mypy to
*opt-out* of static analysis for that module.  This suppresses both errors
without sprinkling ``# type: ignore`` directives throughout the file and keeps
the rest of the suite in full ``--strict`` compliance.
"""

from __future__ import annotations

from types import ModuleType
from typing import List, Union

import importlib
import pytest
import torch

###############################################################################
# Load the module under test                                                  #
###############################################################################

# NOTE: we import via a *string* so that mypy treats the returned object as an
# untyped ``ModuleType`` (see module docstring for details).
cpu_gpu_transfer: ModuleType = importlib.import_module(
    "spectralmc.models.cpu_gpu_transfer"
)

# Import the public alias directly so mypy can “see” it.
from spectralmc.models.cpu_gpu_transfer import TensorTree as _TensorTree  # noqa: E402

###############################################################################
# Minimal stub for `spectralmc.models.torch.Device`                           #
###############################################################################


class DummyDevice:
    """Minimal stand-in for :class:`spectralmc.models.torch.Device` used only
    in unit tests.

    The real ``Device`` hierarchy carries additional semantics that are
    irrelevant for the copying logic exercised here.  Implementing only
    :meth:`to_torch` keeps the stub concise while still satisfying the runtime
    contract expected by :pyfunc:`spectralmc.models.cpu_gpu_transfer.move_tensor_tree`.

    Attributes:
        _dev: Backing :class:`torch.device` created from *torch_dev*.
    """

    def __init__(self, torch_dev: Union[str, torch.device]) -> None:
        self._dev = torch.device(torch_dev)

    def to_torch(self) -> torch.device:  # pragma: no cover
        """Return the underlying :class:`torch.device`."""
        return self._dev


# Ensure the code under test uses the stub.
setattr(cpu_gpu_transfer, "Device", DummyDevice)

###############################################################################
# Helper – flatten nested TensorTree to a list of tensors                     #
###############################################################################


def _flatten_tensors(tree: _TensorTree) -> List[torch.Tensor]:
    """Return every ``torch.Tensor`` leaf in *tree* using pre-order traversal.

    Args:
        tree: Arbitrarily nested combination of tensors, sequences, mappings,
            and scalar leaves.

    Returns:
        A list containing every tensor encountered, in traversal order.  This
        allows the tests to make collective assertions about device placement
        and storage aliases without caring about the original container
        topology.
    """

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
    """Attempting to copy to the current device must raise ``ValueError``."""

    tree: _TensorTree = [torch.ones(2, 3), {"foo": 123}]
    with pytest.raises(ValueError):
        cpu_gpu_transfer.move_tensor_tree(
            tree, dest=DummyDevice("cpu"), pin_memory=False
        )


def test_unsupported_destination_device() -> None:
    """Passing an unknown device string triggers ``RuntimeError``."""

    tensor = torch.zeros(4)
    with pytest.raises(RuntimeError, match="Unsupported destination device"):
        cpu_gpu_transfer.move_tensor_tree(tensor, dest=DummyDevice("xpu"))


def test_cuda_requested_but_not_available(monkeypatch: pytest.MonkeyPatch) -> None:
    """Explicit CUDA request must fail when ``torch.cuda.is_available()`` is ``False``."""

    monkeypatch.setattr(cpu_gpu_transfer.torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA destination requested"):
        cpu_gpu_transfer.move_tensor_tree(
            torch.tensor([1.0]), dest=DummyDevice("cuda:0")
        )


###############################################################################
# GPU-dependent tests – these **must** run; absence of CUDA fails the suite   #
###############################################################################


def _assert_cuda_available() -> None:
    """Abort the test-suite if no CUDA device is detected."""

    if not torch.cuda.is_available():
        pytest.fail("CUDA device required for this test-suite but none detected.")


def test_cpu_to_cuda_and_back_roundtrip() -> None:
    """Round-trip a nested *TensorTree* CPU → CUDA → CPU and verify invariants."""

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
        # Ensure deep-copy (no shared storage).
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
    """Verify that the *pin_memory* flag propagates to CPU allocations."""

    _assert_cuda_available()

    gpu_tensor = torch.randn(8, 8, device="cuda")
    cpu_tensor = cpu_gpu_transfer.move_tensor_tree(
        gpu_tensor, dest=DummyDevice("cpu"), pin_memory=pin_memory
    )
    assert cpu_tensor.device.type == "cpu"
    assert cpu_tensor.is_pinned() == pin_memory
