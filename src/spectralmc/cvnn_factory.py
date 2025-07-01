"""
spectralmc.cvnn_factory
=======================

Declarative, Pydantic‑driven factory for complex‑valued neural networks (CVNNs).

Core ideas
----------
*   **Topology is pure data** – :class:`CVNNConfig` can be JSON‑serialised.
*   Model is *always* built on **CPU** so initialisers (e.g. Xavier) are
    reproducible across machines and GPUs.
*   Checkpoint helpers (`load_model`, `get_safetensors`) interact with
    :class:`~spectralmc.models.torch.TensorState` and never allocate on GPU.
"""

from __future__ import annotations

from enum import Enum
from typing import Dict, Iterator, List, Optional, Tuple, TypeAlias, Union

import torch
import torch.nn as nn
from pydantic import BaseModel, PositiveInt

from spectralmc.models.torch import TensorState, dtype, default_dtype
from spectralmc.cvnn import (  # low‑level building blocks
    ComplexLinear,
    ComplexResidual,
    ComplexSequential,
    CovarianceComplexBatchNorm,
    NaiveComplexBatchNorm,
    modReLU,
    zReLU,
)

__all__: Tuple[str, ...] = (
    "ActivationKind",
    "LayerKind",
    "ActivationCfg",
    "LinearCfg",
    "NaiveBNCfg",
    "CovBNCfg",
    "SequentialCfg",
    "ResidualCfg",
    "CVNNConfig",
    "build_model",
    "load_model",
    "get_safetensors",
)

# ───────────────────────────── enumeration helpers ───────────────────────────


class ActivationKind(str, Enum):
    Z_RELU = "zReLU"
    MOD_RELU = "modReLU"


class LayerKind(str, Enum):
    LINEAR = "ComplexLinear"
    BN_NAIVE = "NaiveComplexBatchNorm"
    BN_COV = "CovarianceComplexBatchNorm"
    SEQ = "Sequential"
    RES = "Residual"


# ─────────────────────────── Pydantic topology schema ────────────────────────


class ActivationCfg(BaseModel):
    kind: ActivationKind


class LinearCfg(BaseModel):
    kind: LayerKind = LayerKind.LINEAR
    width: Optional[int] = None
    bias: bool = True
    activation: Optional[ActivationCfg] = None


class NaiveBNCfg(BaseModel):
    kind: LayerKind = LayerKind.BN_NAIVE
    eps: float = 1e-5
    momentum: float = 0.1
    affine: bool = True
    track_running_stats: bool = True
    activation: Optional[ActivationCfg] = None


class CovBNCfg(BaseModel):
    kind: LayerKind = LayerKind.BN_COV
    eps: float = 1e-5
    momentum: float = 0.1
    affine: bool = True
    track_running_stats: bool = True
    activation: Optional[ActivationCfg] = None


LayerCfg: TypeAlias = Union[
    LinearCfg, NaiveBNCfg, CovBNCfg, "SequentialCfg", "ResidualCfg"
]


class SequentialCfg(BaseModel):
    kind: LayerKind = LayerKind.SEQ
    layers: List[LayerCfg]
    activation: Optional[ActivationCfg] = None


class ResidualCfg(BaseModel):
    kind: LayerKind = LayerKind.RES
    body: SequentialCfg
    projection: Optional[LinearCfg] = None
    activation: Optional[ActivationCfg] = None


class CVNNConfig(BaseModel):
    """
    Entire network topology **plus** RNG seed & dtype – pure data, no behaviour.
    """

    dtype: dtype  # serialisable wrapper
    layers: List[LayerCfg]
    seed: PositiveInt
    final_activation: Optional[ActivationCfg] = None

    def __eq__(self, other: object) -> bool:  # noqa: D401
        return isinstance(other, CVNNConfig) and self.model_dump() == other.model_dump()


# ─────────────────────────── internal helper functions ───────────────────────


def _make_activation(kind: ActivationKind, width: int) -> nn.Module:
    return zReLU() if kind is ActivationKind.Z_RELU else modReLU(width)


def _seq(*mods: nn.Module) -> nn.Module:
    return mods[0] if len(mods) == 1 else ComplexSequential(*mods)


def _maybe_activate(
    module: nn.Module, act: Optional[ActivationCfg], width: int
) -> nn.Module:
    return _seq(module, _make_activation(act.kind, width)) if act else module


def _maybe_project(module: nn.Module, in_w: int, out_w: int) -> Tuple[nn.Module, int]:
    return (
        (module, in_w)
        if in_w == out_w
        else (_seq(module, ComplexLinear(in_w, out_w)), out_w)
    )


# ───────────────────────── recursive layer builder  ──────────────────────────


def _build_from_cfg(cfg: LayerCfg, cur_w: int) -> Tuple[nn.Module, int]:
    match cfg:

        case LinearCfg() as c:
            out_w = c.width or cur_w
            lyr = ComplexLinear(cur_w, out_w, bias=c.bias)
            return _maybe_activate(lyr, c.activation, out_w), out_w

        case NaiveBNCfg() as c:
            bn = NaiveComplexBatchNorm(
                cur_w,
                eps=c.eps,
                momentum=c.momentum,
                affine=c.affine,
                track_running_stats=c.track_running_stats,
            )
            return _maybe_activate(bn, c.activation, cur_w), cur_w

        case CovBNCfg() as c:
            bn = CovarianceComplexBatchNorm(
                cur_w,
                eps=c.eps,
                momentum=c.momentum,
                affine=c.affine,
                track_running_stats=c.track_running_stats,
            )
            return _maybe_activate(bn, c.activation, cur_w), cur_w

        case SequentialCfg() as c:

            def _fold(lst: List[LayerCfg], w_in: int) -> Tuple[List[nn.Module], int]:
                if not lst:
                    return [], w_in
                head, *tail = lst
                head_mod, w_mid = _build_from_cfg(head, w_in)
                tail_mods, w_out = _fold(tail, w_mid)
                return [head_mod, *tail_mods], w_out

            mods, width = _fold(c.layers, cur_w)
            seq = _seq(*mods)
            return _maybe_activate(seq, c.activation, width), width

        case ResidualCfg() as c:
            body_mod, body_w = _build_from_cfg(c.body, cur_w)
            proj_mod, proj_w = (
                _build_from_cfg(c.projection, cur_w)
                if c.projection is not None
                else (
                    (None, body_w)
                    if body_w == cur_w
                    else (ComplexLinear(cur_w, body_w), body_w)
                )
            )
            assert proj_w == body_w, "Projection width mismatch in Residual block"
            res = ComplexResidual(
                body=body_mod,
                proj=proj_mod,
                post_act=(
                    _make_activation(c.activation.kind, body_w)
                    if c.activation
                    else None
                ),
            )
            return res, body_w

    raise RuntimeError(f"Unhandled cfg node: {type(cfg).__name__}")


# ───────────────────────────── public factory api ────────────────────────────


def build_model(
    *,
    n_inputs: int,
    n_outputs: int,
    cfg: CVNNConfig,
) -> nn.Module:
    """
    Instantiate a CVNN **on CPU** with reproducible Xavier initialisation.
    """
    torch.manual_seed(cfg.seed)
    torch_dtype = cfg.dtype.to_torch()

    with default_dtype(torch_dtype):
        body, width = _build_from_cfg(SequentialCfg(layers=cfg.layers), n_inputs)
        body, width = _maybe_project(body, width, n_outputs)
        net = _maybe_activate(body, cfg.final_activation, width)

    return net


def load_model(
    *,
    n_inputs: int,
    n_outputs: int,
    cfg: CVNNConfig,
    tensors: Dict[str, TensorState],
) -> nn.Module:
    """
    Build the model on **CPU** and restore parameters/buffers from *tensors*.
    """
    net = build_model(n_inputs=n_inputs, n_outputs=n_outputs, cfg=cfg)
    state_dict = {
        k: ts.to_tensor(device=torch.device("cpu")) for k, ts in tensors.items()
    }
    net.load_state_dict(state_dict, assign=True)  # zero‑copy ownership
    return net


def get_safetensors(model: nn.Module) -> Dict[str, TensorState]:
    """
    Snapshot *model* into a ``name → TensorState`` dict.

    Raises
    ------
    RuntimeError
        If any parameter/buffer is not resident on CPU.
    """
    if any(p.device.type != "cpu" for p in model.parameters()):
        raise RuntimeError("Model must reside on CPU before serialisation.")

    return {
        name: TensorState.from_tensor(tensor)
        for name, tensor in model.state_dict().items()
    }
