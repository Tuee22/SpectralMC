"""Factory helpers for building and loading complex-valued neural networks."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TypeAlias

from pydantic import BaseModel, ConfigDict, PositiveInt, ValidationError

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
)
from spectralmc.models.torch import (
    AnyDType,
    Device,
    TensorState,
    default_device,
    default_dtype,
)
from spectralmc.result import Failure, Result, Success, collect_results, fold_results
from spectralmc.validation import validate_model
import torch
from spectralmc.runtime import get_torch_handle


get_torch_handle()
nn = torch.nn

__all__: tuple[str, ...] = (
    "ActivationKind",
    "LayerKind",
    "WidthSpec",
    "PreserveWidth",
    "ExplicitWidth",
    "CVNNConfig",
    "build_cvnn_config",
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


LayerCfg: TypeAlias = LinearCfg | NaiveBNCfg | CovBNCfg | SequentialCfg | ResidualCfg


class CVNNConfig(BaseModel):
    dtype: AnyDType
    layers: list[LayerCfg]
    seed: PositiveInt
    final_activation: ActivationCfg | None = None

    model_config = ConfigDict(frozen=True, extra="forbid")


def build_cvnn_config(
    *,
    dtype: AnyDType,
    layers: list[LayerCfg],
    seed: int,
    final_activation: ActivationCfg | None = None,
) -> Result[CVNNConfig, ValidationError]:
    """Result-wrapped constructor for CVNNConfig."""
    return validate_model(
        CVNNConfig,
        dtype=dtype,
        layers=layers,
        seed=seed,
        final_activation=final_activation,
    )


# ─────────────────────────── builder helpers ───────────────────────────────
def _make_activation(kind: ActivationKind, width: int) -> nn.Module:
    return zReLU() if kind is ActivationKind.Z_RELU else modReLU(width)


def _seq(*mods: nn.Module) -> nn.Module:
    return mods[0] if len(mods) == 1 else ComplexSequential(*mods)


def _maybe_activate(mod: nn.Module, act: ActivationCfg | None, width: int) -> nn.Module:
    return _seq(mod, _make_activation(act.kind, width)) if act else mod


def _maybe_project(mod: nn.Module, in_w: int, out_w: int) -> tuple[nn.Module, int]:
    return (mod, in_w) if in_w == out_w else (_seq(mod, ComplexLinear(in_w, out_w)), out_w)


@dataclass(frozen=True)
class _LayerBuildState:
    """Immutable state for layer sequence building."""

    modules: tuple[nn.Module, ...]
    width: int


def _build_single_layer(
    state: _LayerBuildState,
    layer: LayerCfg,
) -> CVNNFactoryResult[_LayerBuildState]:
    """Build a single layer, returning new immutable state or propagating error."""
    build_result = _build_from_cfg(layer, state.width)

    # Pattern match to safely extract tuple without using .value
    result: CVNNFactoryResult[_LayerBuildState]
    match build_result:
        case Success((built_mod, next_w)):
            result = Success(_LayerBuildState(modules=state.modules + (built_mod,), width=next_w))
        case Failure(error):
            result = Failure(error)
    return result


def _build_layer_sequence(
    layers: list[LayerCfg],
    init_w: int,
) -> CVNNFactoryResult[tuple[list[nn.Module], int]]:
    initial_state = _LayerBuildState(modules=(), width=init_w)
    final_state_result = fold_results(layers, _build_single_layer, initial_state)

    match final_state_result:
        case Failure(error):
            return Failure(error)
        case Success(final_state):
            return Success((list(final_state.modules), final_state.width))


def _build_projection(
    projection_cfg: LinearCfg | None,
    cur_w: int,
    body_w: int,
) -> CVNNFactoryResult[tuple[nn.Module | None, int]]:
    """Build projection module for residual connection.

    Returns (projection_module, output_width) or Failure if projection invalid.

    Three cases:
    1. User specified projection: Use it
    2. Widths match, no projection needed: Return (None, body_w)
    3. Widths differ, auto-create: Return (ComplexLinear(cur_w, body_w), body_w)
    """
    match projection_cfg:
        case None if body_w == cur_w:
            # No projection needed, widths already match
            return Success((None, body_w))
        case None:
            # Auto-create projection to match widths
            return Success((ComplexLinear(cur_w, body_w), body_w))
        case LinearCfg() as proj:
            # User-specified projection (explicit type widening: Module -> Module | None)
            match _build_from_cfg(proj, cur_w):
                case Success((module, width)):
                    return Success((module, width))
                case Failure(error):
                    return Failure(error)
                case _:
                    raise AssertionError("Unreachable: Result must be Success or Failure")
        case _:
            raise AssertionError("Unreachable: all projection_cfg cases handled")


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

            # Use helper to build projection
            match _build_projection(c.projection, cur_w, body_w):
                case Failure(error):
                    return Failure(error)
                case Success((proj_mod, proj_w)):
                    pass

            # Validate widths match
            match proj_w == body_w:
                case False:
                    return Failure(
                        SerializationDeviceMismatch(
                            message=f"Residual projection width {proj_w} does not match body width {body_w}."
                        )
                    )
                case True:
                    pass  # Widths match, continue

            res = ComplexResidual(
                body=body_mod,
                proj=proj_mod,
                post_act=_make_activation(c.activation.kind, body_w) if c.activation else None,
            )
            return Success((res, body_w))
    # Unreachable: all LayerCfg variants handled above
    # Note: mypy match exhaustiveness checking insufficient; requires explicit termination
    raise AssertionError(f"Unreachable: unexpected LayerCfg variant {type(cfg).__name__}")


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

        match _build_layer_sequence(cfg.layers, n_inputs):
            case Failure(error):
                return Failure(error)
            case Success((mods, width)):
                seq_mod = _seq(*mods)
                body, width = _maybe_project(seq_mod, width, n_outputs)
                net = _maybe_activate(body, cfg.final_activation, width)
        return Success(net)


def _convert_tensor_state_entry(
    name_ts: tuple[str, TensorState],
) -> CVNNFactoryResult[tuple[str, torch.Tensor]]:
    """Convert single TensorState entry to PyTorch tensor."""
    name, ts = name_ts
    match ts.to_torch():
        case Failure(error):
            return Failure(error)
        case Success(tensor):
            return Success((name, tensor))


def load_model(
    *, model: nn.Module, tensors: dict[str, TensorState]
) -> CVNNFactoryResult[nn.Module]:
    off_cpu = next((p for p in model.parameters() if p.device.type != Device.cpu.value), None)
    match off_cpu:
        case None:
            pass  # All parameters on CPU, continue
        case param:
            return Failure(ModelOnWrongDevice(device=str(param.device)))

    tensor_results = [_convert_tensor_state_entry((name, ts)) for name, ts in tensors.items()]

    match collect_results(tensor_results):
        case Failure(error):
            return Failure(error)
        case Success(entries):
            state_dict = dict(entries)
            model.load_state_dict(state_dict, assign=True)
            return Success(model)


def _convert_torch_tensor_entry(
    name_tensor: tuple[str, torch.Tensor],
) -> CVNNFactoryResult[tuple[str, TensorState]]:
    """Convert PyTorch tensor to TensorState entry."""
    name, tensor = name_tensor
    match TensorState.from_torch(tensor):
        case Failure(error):
            return Failure(error)
        case Success(state):
            return Success((name, state))


def get_safetensors(model: nn.Module) -> CVNNFactoryResult[dict[str, TensorState]]:
    off_cpu = next((p for p in model.parameters() if p.device.type != Device.cpu.value), None)
    match off_cpu:
        case None:
            pass  # All parameters on CPU, continue
        case param:
            return Failure(ModelOnWrongDevice(device=str(param.device)))

    tensor_results = [
        _convert_torch_tensor_entry((name, tensor)) for name, tensor in model.state_dict().items()
    ]

    match collect_results(tensor_results):
        case Failure(error):
            return Failure(error)
        case Success(entries):
            return Success(dict(entries))
