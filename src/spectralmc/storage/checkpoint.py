# src/spectralmc/storage/checkpoint.py
"""Utilities for creating and managing model checkpoints."""

from __future__ import annotations

from typing import Optional

# CRITICAL: Import facade BEFORE torch for deterministic algorithms
import spectralmc.models.torch as sm_torch  # noqa: E402
import torch  # noqa: E402

from spectralmc.gbm_trainer import GbmCVNNPricerConfig, ComplexValuedModel  # noqa: E402
from spectralmc.models.torch import AdamOptimizerState
from spectralmc.serialization.tensors import ModelCheckpointConverter
from spectralmc.serialization import compute_sha256
from spectralmc.storage.store import AsyncBlockchainModelStore
from spectralmc.storage.chain import ModelVersion


def create_checkpoint_from_snapshot(
    snapshot: GbmCVNNPricerConfig,
) -> tuple[bytes, str]:
    """
    Create a serialized checkpoint from a training snapshot.

    Args:
        snapshot: GbmCVNNPricerConfig from GbmCVNNPricer.snapshot()

    Returns:
        Tuple of (checkpoint_bytes, content_hash)
    """
    # Extract model state dict from CVNN
    model_state_dict = snapshot.cvnn.state_dict()

    # Get optimizer state (may be None)
    optimizer_state = snapshot.optimizer_state
    if optimizer_state is None:
        # Create empty optimizer state for serialization
        optimizer_state = AdamOptimizerState(param_states={}, param_groups=[])

    # Get RNG states
    cpu_rng_state = snapshot.torch_cpu_rng_state or b""
    cuda_rng_states = snapshot.torch_cuda_rng_states or []

    # Global step
    global_step = snapshot.global_step

    # Serialize to protobuf
    checkpoint_proto = ModelCheckpointConverter.to_proto(
        model_state_dict=model_state_dict,
        optimizer_state=optimizer_state,
        torch_cpu_rng_state=cpu_rng_state,
        torch_cuda_rng_states=cuda_rng_states,
        global_step=global_step,
    )

    # Serialize to bytes
    checkpoint_bytes = checkpoint_proto.SerializeToString()

    # Compute content hash
    content_hash = compute_sha256(checkpoint_bytes)

    return checkpoint_bytes, content_hash


async def commit_snapshot(
    store: AsyncBlockchainModelStore,
    snapshot: GbmCVNNPricerConfig,
    message: str = "",
) -> ModelVersion:
    """
    Commit a training snapshot to the blockchain store.

    Args:
        store: AsyncBlockchainModelStore instance
        snapshot: GbmCVNNPricerConfig from GbmCVNNPricer.snapshot()
        message: Optional commit message

    Returns:
        ModelVersion for the committed checkpoint
    """
    checkpoint_bytes, content_hash = create_checkpoint_from_snapshot(snapshot)

    # Commit to blockchain
    version = await store.commit(
        checkpoint_data=checkpoint_bytes,
        content_hash=content_hash,
        message=message,
    )

    return version


async def load_snapshot_from_checkpoint(
    store: AsyncBlockchainModelStore,
    version: ModelVersion,
    cvnn_template: torch.nn.Module,
    cfg: GbmCVNNPricerConfig,
) -> GbmCVNNPricerConfig:
    """
    Load a snapshot from a blockchain checkpoint.

    Args:
        store: AsyncBlockchainModelStore instance
        version: ModelVersion to load
        cvnn_template: Empty CVNN model to load state into
        cfg: Base GbmCVNNPricerConfig (only cfg field is used, rest is loaded)

    Returns:
        Reconstructed GbmCVNNPricerConfig
    """
    # Load checkpoint bytes
    checkpoint_bytes = await store.load_checkpoint(version)

    # Deserialize protobuf
    from spectralmc.proto import tensors_pb2

    checkpoint_proto = tensors_pb2.ModelCheckpointProto()
    checkpoint_proto.ParseFromString(checkpoint_bytes)

    # Deserialize checkpoint components
    model_state_dict, optimizer_state, cpu_rng_state, cuda_rng_states, global_step = (
        ModelCheckpointConverter.from_proto(checkpoint_proto)
    )

    # Load model state dict into CVNN
    cvnn_template.load_state_dict(model_state_dict)

    # Validate that cvnn_template implements ComplexValuedModel protocol
    if not isinstance(cvnn_template, ComplexValuedModel):
        raise TypeError(
            f"cvnn_template must implement ComplexValuedModel protocol, "
            f"got {type(cvnn_template).__name__}"
        )
    cvnn: ComplexValuedModel = cvnn_template

    # Create GbmCVNNPricerConfig
    return GbmCVNNPricerConfig(
        cfg=cfg.cfg,  # Use existing BlackScholes config
        domain_bounds=cfg.domain_bounds,  # Use existing domain bounds
        cvnn=cvnn,
        optimizer_state=optimizer_state if optimizer_state.param_states else None,
        global_step=global_step,
        sobol_skip=cfg.sobol_skip,  # Use existing sobol skip
        torch_cpu_rng_state=cpu_rng_state if cpu_rng_state else None,
        torch_cuda_rng_states=cuda_rng_states if cuda_rng_states else None,
    )


__all__ = [
    "create_checkpoint_from_snapshot",
    "commit_snapshot",
    "load_snapshot_from_checkpoint",
]
