"""
Tests for Effect ADT types.

Verifies frozen dataclass behavior, __post_init__ validation,
and pattern matching exhaustiveness.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from spectralmc.effects import (
    BackwardPass,
    CaptureRNGState,
    CommitVersion,
    ComputeFFT,
    Effect,
    ForwardPass,
    GenerateNormals,
    GPUEffect,
    KernelLaunch,
    MonteCarloEffect,
    OptimizerStep,
    ReadObject,
    RestoreRNGState,
    RNGEffect,
    SimulatePaths,
    StorageEffect,
    StreamSync,
    TensorTransfer,
    TrainingEffect,
    WriteObject,
)
from spectralmc.models.torch import Device


class TestGPUEffects:
    """Tests for GPU effect ADTs."""

    def test_tensor_transfer_creation(self) -> None:
        """TensorTransfer is created with correct defaults."""
        effect = TensorTransfer(tensor_id="weights")
        assert effect.kind == "TensorTransfer"
        assert effect.source_device == Device.cuda
        assert effect.target_device == Device.cpu
        assert effect.tensor_id == "weights"

    def test_tensor_transfer_frozen(self) -> None:
        """TensorTransfer is immutable."""
        effect = TensorTransfer(tensor_id="weights")
        with pytest.raises(FrozenInstanceError):
            setattr(effect, "tensor_id", "other")

    def test_tensor_transfer_same_device_raises(self) -> None:
        """TensorTransfer raises ValueError for same source and target device."""
        with pytest.raises(ValueError, match="source and target are both"):
            TensorTransfer(source_device=Device.cpu, target_device=Device.cpu, tensor_id="x")

    def test_stream_sync_creation(self) -> None:
        """StreamSync is created with correct defaults."""
        effect = StreamSync()
        assert effect.kind == "StreamSync"
        assert effect.stream_type == "torch"

    def test_stream_sync_types(self) -> None:
        """StreamSync can be created with different stream types."""
        for stream_type in ("torch", "cupy", "numba"):
            effect = StreamSync(stream_type=stream_type)
            assert effect.stream_type == stream_type

    def test_kernel_launch_creation(self) -> None:
        """KernelLaunch is created with correct values."""
        effect = KernelLaunch(
            kernel_name="simulate_gbm",
            grid_config=(256,),
            block_config=(128,),
        )
        assert effect.kind == "KernelLaunch"
        assert effect.kernel_name == "simulate_gbm"
        assert effect.grid_config == (256,)
        assert effect.block_config == (128,)

    def test_gpu_effect_union(self) -> None:
        """All GPU effect types are part of the GPUEffect union."""
        effects: list[GPUEffect] = [
            TensorTransfer(tensor_id="x"),
            StreamSync(),
            KernelLaunch(kernel_name="test"),
        ]
        assert len(effects) == 3


class TestTrainingEffects:
    """Tests for Training effect ADTs."""

    def test_forward_pass_creation(self) -> None:
        """ForwardPass is created with correct values."""
        effect = ForwardPass(model_id="cvnn", input_tensor_id="batch")
        assert effect.kind == "ForwardPass"
        assert effect.model_id == "cvnn"
        assert effect.input_tensor_id == "batch"

    def test_backward_pass_creation(self) -> None:
        """BackwardPass is created with correct values."""
        effect = BackwardPass(loss_tensor_id="loss")
        assert effect.kind == "BackwardPass"
        assert effect.loss_tensor_id == "loss"

    def test_optimizer_step_creation(self) -> None:
        """OptimizerStep is created with correct values."""
        effect = OptimizerStep(optimizer_id="adam")
        assert effect.kind == "OptimizerStep"
        assert effect.optimizer_id == "adam"

    def test_training_effect_union(self) -> None:
        """All Training effect types are part of the TrainingEffect union."""
        effects: list[TrainingEffect] = [
            ForwardPass(model_id="m"),
            BackwardPass(loss_tensor_id="l"),
            OptimizerStep(optimizer_id="o"),
        ]
        assert len(effects) == 3


class TestMonteCarloEffects:
    """Tests for Monte Carlo effect ADTs."""

    def test_generate_normals_creation(self) -> None:
        """GenerateNormals is created with correct values."""
        effect = GenerateNormals(rows=1024, cols=252, seed=42, skip=0)
        assert effect.kind == "GenerateNormals"
        assert effect.rows == 1024
        assert effect.cols == 252
        assert effect.seed == 42
        assert effect.skip == 0

    def test_simulate_paths_defaults(self) -> None:
        """SimulatePaths has sensible defaults."""
        effect = SimulatePaths()
        assert effect.kind == "SimulatePaths"
        assert effect.spot == 100.0
        assert effect.strike == 100.0
        assert effect.rate == 0.05
        assert effect.vol == 0.2
        assert effect.expiry == 1.0
        assert effect.timesteps == 252
        assert effect.batches == 1024

    def test_compute_fft_creation(self) -> None:
        """ComputeFFT is created with correct values."""
        effect = ComputeFFT(input_tensor_id="paths", axis=-1)
        assert effect.kind == "ComputeFFT"
        assert effect.input_tensor_id == "paths"
        assert effect.axis == -1

    def test_montecarlo_effect_union(self) -> None:
        """All MonteCarlo effect types are part of the MonteCarloEffect union."""
        effects: list[MonteCarloEffect] = [
            GenerateNormals(rows=10, cols=10),
            SimulatePaths(),
            ComputeFFT(input_tensor_id="x"),
        ]
        assert len(effects) == 3


class TestStorageEffects:
    """Tests for Storage effect ADTs."""

    def test_read_object_creation(self) -> None:
        """ReadObject is created with correct values."""
        effect = ReadObject(bucket="models", key="v1/checkpoint.pb")
        assert effect.kind == "ReadObject"
        assert effect.bucket == "models"
        assert effect.key == "v1/checkpoint.pb"

    def test_write_object_creation(self) -> None:
        """WriteObject is created with correct values."""
        effect = WriteObject(bucket="models", key="v1/checkpoint.pb", content_hash="abc123")
        assert effect.kind == "WriteObject"
        assert effect.bucket == "models"
        assert effect.key == "v1/checkpoint.pb"
        assert effect.content_hash == "abc123"

    def test_commit_version_creation(self) -> None:
        """CommitVersion is created with correct values."""
        effect = CommitVersion(
            parent_counter=42, checkpoint_hash="hash123", message="Training complete"
        )
        assert effect.kind == "CommitVersion"
        assert effect.parent_counter == 42
        assert effect.checkpoint_hash == "hash123"
        assert effect.message == "Training complete"

    def test_commit_version_genesis(self) -> None:
        """CommitVersion can be created as genesis (no parent)."""
        effect = CommitVersion(parent_counter=None, checkpoint_hash="hash", message="Genesis")
        assert effect.parent_counter is None

    def test_storage_effect_union(self) -> None:
        """All Storage effect types are part of the StorageEffect union."""
        effects: list[StorageEffect] = [
            ReadObject(bucket="b", key="k"),
            WriteObject(bucket="b", key="k"),
            CommitVersion(checkpoint_hash="h"),
        ]
        assert len(effects) == 3


class TestRNGEffects:
    """Tests for RNG effect ADTs."""

    def test_capture_rng_state_creation(self) -> None:
        """CaptureRNGState is created with correct values."""
        effect = CaptureRNGState(rng_type="torch_cpu")
        assert effect.kind == "CaptureRNGState"
        assert effect.rng_type == "torch_cpu"

    def test_capture_rng_state_types(self) -> None:
        """CaptureRNGState can be created with different RNG types."""
        for rng_type in ("torch_cpu", "torch_cuda", "cupy", "numpy"):
            effect = CaptureRNGState(rng_type=rng_type)
            assert effect.rng_type == rng_type

    def test_restore_rng_state_creation(self) -> None:
        """RestoreRNGState is created with correct values."""
        state_bytes = b"\x00\x01\x02\x03"
        effect = RestoreRNGState(rng_type="torch_cpu", state_bytes=state_bytes)
        assert effect.kind == "RestoreRNGState"
        assert effect.rng_type == "torch_cpu"
        assert effect.state_bytes == state_bytes

    def test_rng_effect_union(self) -> None:
        """All RNG effect types are part of the RNGEffect union."""
        effects: list[RNGEffect] = [
            CaptureRNGState(),
            RestoreRNGState(),
        ]
        assert len(effects) == 2


class TestEffectUnion:
    """Tests for the master Effect union."""

    def test_all_effects_in_master_union(self) -> None:
        """All effect types are part of the master Effect union."""
        effects: list[Effect] = [
            # GPU
            TensorTransfer(tensor_id="x"),
            StreamSync(),
            KernelLaunch(kernel_name="k"),
            # Training
            ForwardPass(model_id="m"),
            BackwardPass(loss_tensor_id="l"),
            OptimizerStep(optimizer_id="o"),
            # Monte Carlo
            GenerateNormals(rows=10, cols=10),
            SimulatePaths(),
            ComputeFFT(input_tensor_id="t"),
            # Storage
            ReadObject(bucket="b", key="k"),
            WriteObject(bucket="b", key="k"),
            CommitVersion(checkpoint_hash="h"),
            # RNG
            CaptureRNGState(),
            RestoreRNGState(),
        ]
        assert len(effects) == 14

    def test_pattern_matching_gpu_effects(self) -> None:
        """Pattern matching works for GPU effects."""
        effect: Effect = TensorTransfer(
            source_device=Device.cuda, target_device=Device.cpu, tensor_id="w"
        )
        match effect:
            case TensorTransfer(tensor_id=tid):
                assert tid == "w"
            case _:
                pytest.fail("Should have matched TensorTransfer")

    def test_pattern_matching_training_effects(self) -> None:
        """Pattern matching works for training effects."""
        effect: Effect = ForwardPass(model_id="cvnn", input_tensor_id="input")
        match effect:
            case ForwardPass(model_id=mid, input_tensor_id=tid):
                assert mid == "cvnn"
                assert tid == "input"
            case _:
                pytest.fail("Should have matched ForwardPass")
