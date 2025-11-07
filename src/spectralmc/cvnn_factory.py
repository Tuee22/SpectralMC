# src/spectralmc/cvnn_factory.py
from __future__ import annotations

from enum import Enum
from typing import Dict, List, Optional, Tuple, TypeAlias, Union

import torch
import torch.nn as nn
from pydantic import BaseModel, ConfigDict, PositiveInt

from spectralmc.models.torch import (
    TensorState,
    DType,
    Device,
    default_dtype,
    default_device,
)
from spectralmc.cvnn import (
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
    "WidthSpec",
    "PreserveWidth",
    "ExplicitWidth",
    "CVNNConfig",
    "build_model",
    "load_model",
    "get_safetensors",
)


# ───────────────────────────── enumerations ────────────────────────────────
class ActivationKind(str, Enum):
    Z_RELU = "zReLU"
    MOD_RELU = "modReLU"


class LayerKind(str, Enum):
    LINEAR = "ComplexLinear"
    BN_NAIVE = "NaiveComplexBatchNorm"
    BN_COV = "CovarianceComplexBatchNorm"
    SEQ = "Sequential"
    RES = "Residual"


# ───────────────────────────── config schemas ───────────────────────────────


# Width specification ADT: explicit or preserve input width
class WidthSpec(BaseModel):
    """Base class for width specification."""

    model_config = ConfigDict(frozen=True)


class PreserveWidth(WidthSpec):
    """Tag-only class indicating width should be preserved from input."""

    pass


class ExplicitWidth(WidthSpec):
    """Explicit width specification."""

    value: PositiveInt


class ActivationCfg(BaseModel):
    kind: ActivationKind


class LinearCfg(BaseModel):
    kind: LayerKind = LayerKind.LINEAR
    width: WidthSpec = PreserveWidth()
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
    dtype: DType
    layers: List[LayerCfg]
    seed: PositiveInt
    final_activation: Optional[ActivationCfg] = None


# ─────────────────────────── builder helpers ───────────────────────────────
def _make_activation(kind: ActivationKind, width: int) -> nn.Module:
    return zReLU() if kind is ActivationKind.Z_RELU else modReLU(width)


def _seq(*mods: nn.Module) -> nn.Module:
    return mods[0] if len(mods) == 1 else ComplexSequential(*mods)


def _maybe_activate(
    mod: nn.Module, act: Optional[ActivationCfg], width: int
) -> nn.Module:
    return _seq(mod, _make_activation(act.kind, width)) if act else mod


def _maybe_project(mod: nn.Module, in_w: int, out_w: int) -> Tuple[nn.Module, int]:
    return (
        (mod, in_w) if in_w == out_w else (_seq(mod, ComplexLinear(in_w, out_w)), out_w)
    )


def _build_from_cfg(cfg: LayerCfg, cur_w: int) -> Tuple[nn.Module, int]:
    match cfg:

        case LinearCfg() as c:
            # Pattern match on WidthSpec to determine output width
            match c.width:
                case PreserveWidth():
                    out_w = cur_w
                case ExplicitWidth(value=w):
                    out_w = w
            lyr = ComplexLinear(cur_w, out_w, bias=c.bias)
            return _maybe_activate(lyr, c.activation, out_w), out_w

        case NaiveBNCfg() as c:
            naive_bn = NaiveComplexBatchNorm(
                cur_w,
                eps=c.eps,
                momentum=c.momentum,
                affine=c.affine,
                track_running_stats=c.track_running_stats,
            )
            return _maybe_activate(naive_bn, c.activation, cur_w), cur_w

        case CovBNCfg() as c:
            cov_bn = CovarianceComplexBatchNorm(
                cur_w,
                eps=c.eps,
                momentum=c.momentum,
                affine=c.affine,
                track_running_stats=c.track_running_stats,
            )
            return _maybe_activate(cov_bn, c.activation, cur_w), cur_w

        case SequentialCfg() as c:

            def _fold(lst: List[LayerCfg], w_in: int) -> Tuple[List[nn.Module], int]:
                if not lst:
                    return [], w_in
                head, *tail = lst
                hd_mod, w_mid = _build_from_cfg(head, w_in)
                tl_mods, w_out = _fold(tail, w_mid)
                return [hd_mod, *tl_mods], w_out

            submods, width = _fold(c.layers, cur_w)
            seq = _seq(*submods)
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


# ───────────────────────────── public API ───────────────────────────────────
def build_model(*, n_inputs: int, n_outputs: int, cfg: CVNNConfig) -> nn.Module:
    """Build a CVNN model without mutating global RNG state.

    The RNG state (CPU + all CUDA devices) is snapshotted on entry and
    restored automatically on exit, so calls outside this function see the
    same random stream they had before.
    """

    cpu_dev = Device.cpu.to_torch()
    torch_dtype = cfg.dtype.to_torch()

    # Snapshot RNG state and safely reseed just for this scope.
    # The leftmost context manager (`fork_rng`) enters first, so the manual
    # seed call that follows is completely isolated from the caller.
    with torch.random.fork_rng(), default_device(cpu_dev), default_dtype(torch_dtype):
        torch.manual_seed(cfg.seed)  # deterministic but local to this block

        body, width = _build_from_cfg(SequentialCfg(layers=cfg.layers), n_inputs)
        body, width = _maybe_project(body, width, n_outputs)
        net = _maybe_activate(body, cfg.final_activation, width)

    return net


def load_model(*, model: nn.Module, tensors: Dict[str, TensorState]) -> nn.Module:
    if any(p.device.type != Device.cpu.value for p in model.parameters()):
        raise RuntimeError("`model` must be on CPU before loading weights.")

    state_dict = {k: ts.to_torch() for k, ts in tensors.items()}
    model.load_state_dict(state_dict, assign=True)
    return model


def get_safetensors(model: nn.Module) -> Dict[str, TensorState]:
    if any(p.device.type != Device.cpu.value for p in model.parameters()):
        raise RuntimeError("Model must reside on CPU for serialisation.")

    return {name: TensorState.from_torch(t) for name, t in model.state_dict().items()}
