# src/spectralmc/cvnn_factory.py
from __future__ import annotations

from enum import Enum
from typing import TypeAlias, Union

import torch
from pydantic import BaseModel, ConfigDict, PositiveInt
from torch import nn

# CRITICAL: Import facade BEFORE torch for deterministic algorithms
from spectralmc.cvnn import (
    ComplexLinear,
    ComplexResidual,
    ComplexSequential,
    CovarianceComplexBatchNorm,
    NaiveComplexBatchNorm,
    modReLU,
    zReLU,
)
from spectralmc.errors.cvnn_factory import (
    CVNNFactoryResult,
    ModelOnWrongDevice,
    SerializationDeviceMismatch,
    UnhandledConfigNode,
)
from spectralmc.models.torch import (
    AnyDType,
    Device,
    TensorState,
    default_device,
    default_dtype,
)
from spectralmc.result import Failure, Success


__all__: tuple[str, ...] = (
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

    model_config = ConfigDict(frozen=True, extra="forbid")


class PreserveWidth(WidthSpec):
    """Tag-only class indicating width should be preserved from input."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class ExplicitWidth(WidthSpec):
    """Explicit width specification."""

    value: PositiveInt

    model_config = ConfigDict(frozen=True, extra="forbid")


class ActivationCfg(BaseModel):
    kind: ActivationKind

    model_config = ConfigDict(frozen=True, extra="forbid")


class LinearCfg(BaseModel):
    kind: LayerKind = LayerKind.LINEAR
    width: WidthSpec = PreserveWidth()
    bias: bool = True
    activation: ActivationCfg | None = None

    model_config = ConfigDict(frozen=True, extra="forbid")


class NaiveBNCfg(BaseModel):
    kind: LayerKind = LayerKind.BN_NAIVE
    eps: float = 1e-5
    momentum: float = 0.1
    affine: bool = True
    track_running_stats: bool = True
    activation: ActivationCfg | None = None

    model_config = ConfigDict(frozen=True, extra="forbid")


class CovBNCfg(BaseModel):
    kind: LayerKind = LayerKind.BN_COV
    eps: float = 1e-5
    momentum: float = 0.1
    affine: bool = True
    track_running_stats: bool = True
    activation: ActivationCfg | None = None

    model_config = ConfigDict(frozen=True, extra="forbid")


LayerCfg: TypeAlias = Union[LinearCfg, NaiveBNCfg, CovBNCfg, "SequentialCfg", "ResidualCfg"]


class SequentialCfg(BaseModel):
    kind: LayerKind = LayerKind.SEQ
    layers: list[LayerCfg]
    activation: ActivationCfg | None = None

    model_config = ConfigDict(frozen=True, extra="forbid")


class ResidualCfg(BaseModel):
    kind: LayerKind = LayerKind.RES
    body: SequentialCfg
    projection: LinearCfg | None = None
    activation: ActivationCfg | None = None

    model_config = ConfigDict(frozen=True, extra="forbid")


class CVNNConfig(BaseModel):
    dtype: AnyDType
    layers: list[LayerCfg]
    seed: PositiveInt
    final_activation: ActivationCfg | None = None

    model_config = ConfigDict(frozen=True, extra="forbid")


# ─────────────────────────── builder helpers ───────────────────────────────
def _make_activation(kind: ActivationKind, width: int) -> nn.Module:
    return zReLU() if kind is ActivationKind.Z_RELU else modReLU(width)


def _seq(*mods: nn.Module) -> nn.Module:
    return mods[0] if len(mods) == 1 else ComplexSequential(*mods)


def _maybe_activate(mod: nn.Module, act: ActivationCfg | None, width: int) -> nn.Module:
    return _seq(mod, _make_activation(act.kind, width)) if act else mod


def _maybe_project(mod: nn.Module, in_w: int, out_w: int) -> tuple[nn.Module, int]:
    return (mod, in_w) if in_w == out_w else (_seq(mod, ComplexLinear(in_w, out_w)), out_w)


def _build_layer_sequence(
    layers: list[LayerCfg],
    init_w: int,
) -> CVNNFactoryResult[tuple[list[nn.Module], int]]:
    modules: list[nn.Module] = []
    width = init_w
    for layer in layers:
        match _build_from_cfg(layer, width):
            case Failure(error):
                return Failure(error)
            case Success((built_mod, next_w)):
                modules.append(built_mod)
                width = next_w
    return Success((modules, width))


def _build_from_cfg(cfg: LayerCfg, cur_w: int) -> CVNNFactoryResult[tuple[nn.Module, int]]:
    match cfg:

        case LinearCfg() as c:
            match c.width:
                case PreserveWidth():
                    out_w = cur_w
                case ExplicitWidth(value=w):
                    out_w = w
            lyr = ComplexLinear(cur_w, out_w, bias=c.bias)
            return Success((_maybe_activate(lyr, c.activation, out_w), out_w))

        case NaiveBNCfg() as c:
            naive_bn = NaiveComplexBatchNorm(
                cur_w,
                eps=c.eps,
                momentum=c.momentum,
                affine=c.affine,
                track_running_stats=c.track_running_stats,
            )
            return Success((_maybe_activate(naive_bn, c.activation, cur_w), cur_w))

        case CovBNCfg() as c:
            cov_bn = CovarianceComplexBatchNorm(
                cur_w,
                eps=c.eps,
                momentum=c.momentum,
                affine=c.affine,
                track_running_stats=c.track_running_stats,
            )
            return Success((_maybe_activate(cov_bn, c.activation, cur_w), cur_w))

        case SequentialCfg() as c:
            match _build_layer_sequence(c.layers, cur_w):
                case Failure(error):
                    return Failure(error)
                case Success((submods, width)):
                    seq = _seq(*submods)
                    return Success((_maybe_activate(seq, c.activation, width), width))

        case ResidualCfg() as c:
            match _build_from_cfg(c.body, cur_w):
                case Failure(error):
                    return Failure(error)
                case Success((body_mod, body_w)):
                    pass

            proj_mod: nn.Module | None
            proj_w: int
            if c.projection is not None:
                match _build_from_cfg(c.projection, cur_w):
                    case Failure(error):
                        return Failure(error)
                    case Success((proj_candidate, width)):
                        proj_mod = proj_candidate
                        proj_w = width
            elif body_w == cur_w:
                proj_mod = None
                proj_w = body_w
            else:
                proj_mod = ComplexLinear(cur_w, body_w)
                proj_w = body_w

            if proj_w != body_w:
                return Failure(
                    SerializationDeviceMismatch(
                        message=f"Residual projection width {proj_w} does not match body width {body_w}."
                    )
                )

            res = ComplexResidual(
                body=body_mod,
                proj=proj_mod,
                post_act=_make_activation(c.activation.kind, body_w) if c.activation else None,
            )
            return Success((res, body_w))

        case _:
            return Failure(UnhandledConfigNode(node=type(cfg).__name__))


# ───────────────────────────── public API ───────────────────────────────────
def build_model(*, n_inputs: int, n_outputs: int, cfg: CVNNConfig) -> CVNNFactoryResult[nn.Module]:
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

        match _build_from_cfg(SequentialCfg(layers=cfg.layers), n_inputs):
            case Failure(error):
                return Failure(error)
            case Success((body, width)):
                body, width = _maybe_project(body, width, n_outputs)
                net = _maybe_activate(body, cfg.final_activation, width)
        return Success(net)


def load_model(
    *, model: nn.Module, tensors: dict[str, TensorState]
) -> CVNNFactoryResult[nn.Module]:
    off_cpu = next((p for p in model.parameters() if p.device.type != Device.cpu.value), None)
    if off_cpu is not None:
        return Failure(ModelOnWrongDevice(device=str(off_cpu.device)))

    state_dict: dict[str, torch.Tensor] = {}
    for name, ts in tensors.items():
        match ts.to_torch():
            case Failure(error):
                return Failure(error)
            case Success(tensor):
                state_dict[name] = tensor

    model.load_state_dict(state_dict, assign=True)
    return Success(model)


def get_safetensors(model: nn.Module) -> CVNNFactoryResult[dict[str, TensorState]]:
    off_cpu = next((p for p in model.parameters() if p.device.type != Device.cpu.value), None)
    if off_cpu is not None:
        return Failure(ModelOnWrongDevice(device=str(off_cpu.device)))

    safetensors: dict[str, TensorState] = {}
    for name, tensor in model.state_dict().items():
        match TensorState.from_torch(tensor):
            case Failure(error):
                return Failure(error)
            case Success(state):
                safetensors[name] = state
    return Success(safetensors)
