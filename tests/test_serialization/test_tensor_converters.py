# tests/test_serialization/test_tensor_converters.py
"""Tests for tensor and optimizer state serialization.

All tests require GPU - missing GPU is a hard failure, not a skip.
"""

from __future__ import annotations

import torch
import pytest

# Module-level GPU requirement - test file fails immediately without GPU
assert torch.cuda.is_available(), "CUDA required for SpectralMC tests"

GPU_DEV: torch.device = torch.device("cuda:0")

from spectralmc.models.torch import (
    AdamOptimizerState,
    AdamParamState,
    AdamParamGroup,
    TensorState,
)
from spectralmc.serialization.tensors import (
    TensorStateConverter,
    AdamOptimizerStateConverter,
    RNGStateConverter,
)


def test_tensor_state_round_trip_float32() -> None:
    """Test TensorStateConverter with float32 tensor."""
    original = torch.randn(3, 4, dtype=torch.float32)

    proto = TensorStateConverter.to_proto(original)
    recovered = TensorStateConverter.from_proto(proto)

    assert torch.allclose(original, recovered, rtol=1e-6, atol=1e-9)
    assert recovered.dtype == torch.float32
    assert recovered.shape == original.shape


def test_tensor_state_round_trip_float64() -> None:
    """Test TensorStateConverter with float64 tensor."""
    original = torch.randn(5, 10, dtype=torch.float64)

    proto = TensorStateConverter.to_proto(original)
    recovered = TensorStateConverter.from_proto(proto)

    assert torch.allclose(original, recovered, rtol=1e-12, atol=1e-15)
    assert recovered.dtype == torch.float64


def test_tensor_state_round_trip_complex64() -> None:
    """Test TensorStateConverter with complex64 tensor."""
    real = torch.randn(4, 4, dtype=torch.float32)
    imag = torch.randn(4, 4, dtype=torch.float32)
    original = torch.view_as_complex(torch.stack([real, imag], dim=-1))

    proto = TensorStateConverter.to_proto(original)
    recovered = TensorStateConverter.from_proto(proto)

    assert torch.allclose(original, recovered, rtol=1e-6, atol=1e-9)
    assert recovered.dtype == torch.complex64


def test_tensor_state_round_trip_complex128() -> None:
    """Test TensorStateConverter with complex128 tensor."""
    real = torch.randn(3, 5, dtype=torch.float64)
    imag = torch.randn(3, 5, dtype=torch.float64)
    original = torch.view_as_complex(torch.stack([real, imag], dim=-1))

    proto = TensorStateConverter.to_proto(original)
    recovered = TensorStateConverter.from_proto(proto)

    assert torch.allclose(original, recovered, rtol=1e-12, atol=1e-15)
    assert recovered.dtype == torch.complex128


def test_tensor_state_requires_grad() -> None:
    """Test TensorStateConverter preserves requires_grad."""
    original = torch.randn(3, 3, dtype=torch.float32, requires_grad=True)

    proto = TensorStateConverter.to_proto(original)
    recovered = TensorStateConverter.from_proto(proto)

    assert recovered.requires_grad is True


def test_tensor_state_no_requires_grad() -> None:
    """Test TensorStateConverter with requires_grad=False."""
    original = torch.randn(3, 3, dtype=torch.float32, requires_grad=False)

    proto = TensorStateConverter.to_proto(original)
    recovered = TensorStateConverter.from_proto(proto)

    assert recovered.requires_grad is False


def test_adam_optimizer_state_round_trip() -> None:
    """Test AdamOptimizerStateConverter round-trip."""
    # Create sample optimizer state
    param_states = {
        0: AdamParamState.from_torch(
            {
                "step": 10,
                "exp_avg": torch.randn(5, 5),
                "exp_avg_sq": torch.randn(5, 5),
            }
        ),
        1: AdamParamState.from_torch(
            {
                "step": 10,
                "exp_avg": torch.randn(3, 3),
                "exp_avg_sq": torch.randn(3, 3),
            }
        ),
    }

    param_groups = [
        AdamParamGroup(
            params=[0, 1],
            lr=0.001,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.0,
            amsgrad=False,
        )
    ]

    original = AdamOptimizerState(param_states=param_states, param_groups=param_groups)

    # Round trip
    proto = AdamOptimizerStateConverter.to_proto(original)
    recovered = AdamOptimizerStateConverter.from_proto(proto)

    # Verify param states
    assert len(recovered.param_states) == len(original.param_states)
    for param_id in original.param_states:
        orig_state = original.param_states[param_id]
        rec_state = recovered.param_states[param_id]

        assert rec_state.step == orig_state.step
        assert torch.allclose(
            rec_state.exp_avg.to_torch(),
            orig_state.exp_avg.to_torch(),
            rtol=1e-6,
            atol=1e-9,
        )
        assert torch.allclose(
            rec_state.exp_avg_sq.to_torch(),
            orig_state.exp_avg_sq.to_torch(),
            rtol=1e-6,
            atol=1e-9,
        )

    # Verify param groups
    assert len(recovered.param_groups) == len(original.param_groups)
    for i, (orig_group, rec_group) in enumerate(
        zip(original.param_groups, recovered.param_groups)
    ):
        assert rec_group.lr == orig_group.lr
        assert rec_group.betas == orig_group.betas
        assert rec_group.eps == orig_group.eps
        assert rec_group.weight_decay == orig_group.weight_decay
        assert rec_group.amsgrad == orig_group.amsgrad


def test_rng_state_round_trip() -> None:
    """Test RNGStateConverter round-trip."""
    cpu_state = torch.get_rng_state().numpy().tobytes()
    cuda_states = []

    # GPU is always available (enforced by module-level assertion)
    for i in range(torch.cuda.device_count()):
        cuda_states.append(torch.cuda.get_rng_state(i).numpy().tobytes())

    # Round trip
    proto = RNGStateConverter.to_proto(cpu_state, cuda_states)
    rec_cpu, rec_cuda = RNGStateConverter.from_proto(proto)

    assert rec_cpu == cpu_state
    assert rec_cuda == cuda_states


def test_tensor_empty() -> None:
    """Test TensorStateConverter with empty tensor."""
    original = torch.tensor([], dtype=torch.float32)

    proto = TensorStateConverter.to_proto(original)
    recovered = TensorStateConverter.from_proto(proto)

    assert recovered.shape == original.shape
    assert recovered.dtype == original.dtype


def test_tensor_scalar() -> None:
    """Test TensorStateConverter with scalar tensor."""
    original = torch.tensor(42.0, dtype=torch.float32)

    proto = TensorStateConverter.to_proto(original)
    recovered = TensorStateConverter.from_proto(proto)

    assert torch.allclose(original, recovered)
    assert recovered.shape == original.shape


def test_tensor_large() -> None:
    """Test TensorStateConverter with large tensor."""
    original = torch.randn(100, 100, 100, dtype=torch.float32)

    proto = TensorStateConverter.to_proto(original)
    recovered = TensorStateConverter.from_proto(proto)

    assert torch.allclose(original, recovered, rtol=1e-6, atol=1e-9)
    assert recovered.shape == original.shape


def test_model_checkpoint_round_trip() -> None:
    """Test ModelCheckpointConverter with complete checkpoint."""
    from spectralmc.serialization.tensors import ModelCheckpointConverter

    # Create sample model state dict
    model_state_dict = {
        "layer1.weight": torch.randn(10, 5),
        "layer1.bias": torch.randn(10),
        "layer2.weight": torch.randn(5, 10),
        "layer2.bias": torch.randn(5),
    }

    # Create optimizer state
    param_states = {
        0: AdamParamState.from_torch(
            {
                "step": 100,
                "exp_avg": torch.randn(10, 5),
                "exp_avg_sq": torch.randn(10, 5),
            }
        ),
        1: AdamParamState.from_torch(
            {
                "step": 100,
                "exp_avg": torch.randn(10),
                "exp_avg_sq": torch.randn(10),
            }
        ),
    }

    param_groups = [
        AdamParamGroup(
            params=[0, 1],
            lr=0.001,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.0,
            amsgrad=False,
        )
    ]

    optimizer_state = AdamOptimizerState(
        param_states=param_states, param_groups=param_groups
    )

    # RNG states
    cpu_rng = torch.get_rng_state().numpy().tobytes()
    cuda_rngs: list[bytes] = []

    global_step = 1000

    # Round trip
    proto = ModelCheckpointConverter.to_proto(
        model_state_dict, optimizer_state, cpu_rng, cuda_rngs, global_step
    )

    rec_model, rec_opt, rec_cpu_rng, rec_cuda_rngs, rec_step = (
        ModelCheckpointConverter.from_proto(proto)
    )

    # Verify model state dict
    assert set(rec_model.keys()) == set(model_state_dict.keys())
    for name in model_state_dict:
        assert torch.allclose(
            rec_model[name], model_state_dict[name], rtol=1e-6, atol=1e-9
        )

    # Verify optimizer state
    assert len(rec_opt.param_states) == len(optimizer_state.param_states)

    # Verify RNG state
    assert rec_cpu_rng == cpu_rng
    assert rec_cuda_rngs == cuda_rngs

    # Verify global step
    assert rec_step == global_step
