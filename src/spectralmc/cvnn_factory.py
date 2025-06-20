"""cvnn_factory.py
=================
A *purely-functional* factory for building complex-valued neural networks
(CVNNs) from a nested :class:`CVNNConfig` description.

Highlights
----------
* **Device placement** – parameters are created directly on the requested
  device via ``with torch.device("cuda"):``.
* **Parameter precision** – an internal ``_default_dtype`` context manager
  wraps :pyfunc:`torch.set_default_dtype` so every tensor born inside the scope
  uses the chosen dtype.
* Pure, expression-oriented code – helper combinators replace mutable loops.

The module is free of ``cast``, ``Any`` and ``# type: ignore`` markers.
"""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from enum import Enum
from typing import Iterator, List, Optional, Tuple, TypeAlias, Union

import torch
import torch.nn as nn
from pydantic import BaseModel, PositiveInt

from spectralmc.cvnn import (
    ComplexLinear,
    ComplexResidual,
    ComplexSequential,
    CovarianceComplexBatchNorm,
    NaiveComplexBatchNorm,
    modReLU,
    zReLU,
)

__all__: Tuple[str, ...] = ("CVNNConfig", "build_model")

# =============================================================================
#  Enumerations
# =============================================================================


class ActivationKind(str, Enum):
    """Kinds of complex-domain activation supported by this factory."""

    Z_RELU = "zReLU"
    MOD_RELU = "modReLU"


class LayerKind(str, Enum):
    """Layer primitives that can appear in a :class:`LayerCfg`."""

    LINEAR = "ComplexLinear"
    BN_NAIVE = "NaiveComplexBatchNorm"
    BN_COV = "CovarianceComplexBatchNorm"
    SEQ = "Sequential"
    RES = "Residual"


# =============================================================================
#  Helper combinators (internal use only)
# =============================================================================


def _make_activation(kind: ActivationKind, width: int) -> nn.Module:  # noqa: D401
    """Return a freshly constructed activation module for *kind*."""
    return zReLU() if kind is ActivationKind.Z_RELU else modReLU(width)


def _seq(*mods: nn.Module) -> nn.Module:  # noqa: D401
    """Return *mods* unchanged or wrapped in :class:`ComplexSequential`."""
    return mods[0] if len(mods) == 1 else ComplexSequential(*mods)


def _maybe_activate(
    module: nn.Module,
    act: Optional[ActivationCfg],
    width: int,
) -> nn.Module:
    """Append an activation to *module* when *act* is provided."""
    return _seq(module, _make_activation(act.kind, width)) if act else module


def _maybe_project(
    module: nn.Module,
    in_w: int,
    out_w: int,
) -> Tuple[nn.Module, int]:
    """Insert a projection layer when input/output widths differ."""
    if in_w == out_w:
        return module, in_w
    return _seq(module, ComplexLinear(in_w, out_w)), out_w


# =============================================================================
#  Scoped default-dtype helper
# =============================================================================


@contextmanager
def _default_dtype(dtype: torch.dtype) -> Iterator[None]:
    """Temporarily set the global default dtype for tensor construction."""
    previous = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(previous)


# =============================================================================
#  Pydantic configs – pure data (no behaviour)
# =============================================================================


class ActivationCfg(BaseModel):
    """Config for a complex activation layer."""

    kind: ActivationKind


class LinearCfg(BaseModel):
    """Config for :class:`~spectralmc.cvnn.ComplexLinear`."""

    kind: LayerKind = LayerKind.LINEAR
    width: Optional[int] = None
    bias: bool = True
    activation: Optional[ActivationCfg] = None


class NaiveBNCfg(BaseModel):
    """Config for :class:`~spectralmc.cvnn.NaiveComplexBatchNorm`."""

    kind: LayerKind = LayerKind.BN_NAIVE
    eps: float = 1e-5
    momentum: float = 0.1
    affine: bool = True
    track_running_stats: bool = True
    activation: Optional[ActivationCfg] = None


class CovBNCfg(BaseModel):
    """Config for :class:`~spectralmc.cvnn.CovarianceComplexBatchNorm`."""

    kind: LayerKind = LayerKind.BN_COV
    eps: float = 1e-5
    momentum: float = 0.1
    affine: bool = True
    track_running_stats: bool = True
    activation: Optional[ActivationCfg] = None


LayerCfg: TypeAlias = Union[
    "LinearCfg", "NaiveBNCfg", "CovBNCfg", "SequentialCfg", "ResidualCfg"
]


class SequentialCfg(BaseModel):
    """Config for an ordered container of *layers*."""

    kind: LayerKind = LayerKind.SEQ
    layers: List[LayerCfg]
    activation: Optional[ActivationCfg] = None


class ResidualCfg(BaseModel):
    """Config for a skip-connection *Residual* block."""

    kind: LayerKind = LayerKind.RES
    body: SequentialCfg
    projection: Optional[LinearCfg] = None
    activation: Optional[ActivationCfg] = None


class CVNNConfig(BaseModel):
    """Top-level network specification consumed by :func:`build_model`."""

    layers: List[LayerCfg]
    seed: PositiveInt
    final_activation: Optional[ActivationCfg] = None


# =============================================================================
#  Recursive builder
# =============================================================================


def _build_from_cfg(cfg: LayerCfg, cur_w: int) -> Tuple[nn.Module, int]:
    """Recursively translate *cfg* to an :class:`nn.Module`."""
    match cfg:
        # ── Linear ────────────────────────────────────────────────────────
        case LinearCfg() as c:
            out_w = c.width or cur_w
            lin = ComplexLinear(cur_w, out_w, bias=c.bias)
            return _maybe_activate(lin, c.activation, out_w), out_w

        # ── Naive BN ──────────────────────────────────────────────────────
        case NaiveBNCfg() as c:
            nbn = NaiveComplexBatchNorm(
                num_features=cur_w,
                eps=c.eps,
                momentum=c.momentum,
                affine=c.affine,
                track_running_stats=c.track_running_stats,
            )
            return _maybe_activate(nbn, c.activation, cur_w), cur_w

        case CovBNCfg() as c:
            cbn = CovarianceComplexBatchNorm(
                num_features=cur_w,
                eps=c.eps,
                momentum=c.momentum,
                affine=c.affine,
                track_running_stats=c.track_running_stats,
            )
            return _maybe_activate(cbn, c.activation, cur_w), cur_w

        # ── Sequential ───────────────────────────────────────────────────
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

        # ── Residual ─────────────────────────────────────────────────────
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

    raise RuntimeError(f"Unhandled layer config type: {type(cfg).__name__}")


# =============================================================================
#  Public factory
# =============================================================================


def build_model(
    *,
    n_inputs: int,
    n_outputs: int,
    cfg: CVNNConfig,
    device: Union[str, torch.device, None] = None,
    dtype: Optional[torch.dtype] = None,
) -> nn.Module:
    """Materialise a CVNN described by *cfg* on the requested device and dtype."""

    torch.manual_seed(int(cfg.seed))

    def _materialise() -> nn.Module:
        body, w = _build_from_cfg(SequentialCfg(layers=cfg.layers), n_inputs)
        body, w = _maybe_project(body, w, n_outputs)
        return _maybe_activate(body, cfg.final_activation, w)

    dev_ctx = torch.device(device) if device is not None else nullcontext()
    dt_ctx = _default_dtype(dtype) if dtype is not None else nullcontext()

    with dev_ctx, dt_ctx:
        return _materialise()
