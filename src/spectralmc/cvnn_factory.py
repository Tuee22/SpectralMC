# src/spectralmc/cvnn_factory.py
"""
cvnn_factory.py
===============

Pure, expression‑oriented factory that materialises complex‑valued neural
networks (**CVNNs**) from a declarative :class:`CVNNConfig`.

* Uses :pyfunc:`torch.set_default_device` to create parameters **directly on
  the target GPU** (zero copy) while falling back gracefully on older
  runtimes.
* Import‑time switches enforce full determinism (CuBLAS, CuDNN, Torch).
* The entire module is mypy‑clean under ``--strict``.
"""

from __future__ import annotations

import os
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

# ────────────────────────── reproducibility at import ──────────────────────
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")
torch.use_deterministic_algorithms(True, warn_only=False)
if hasattr(torch.backends, "cudnn"):
    torch.backends.cudnn.deterministic = True  # noqa: PGH003
    torch.backends.cudnn.benchmark = False  # noqa: PGH003

# =============================================================================
#  Enumerations
# =============================================================================


class ActivationKind(str, Enum):
    """Kinds of complex‑domain activation supported by the factory."""

    Z_RELU = "zReLU"
    MOD_RELU = "modReLU"


class LayerKind(str, Enum):
    """Primitive layer families that can appear in a :class:`LayerCfg`."""

    LINEAR = "ComplexLinear"
    BN_NAIVE = "NaiveComplexBatchNorm"
    BN_COV = "CovarianceComplexBatchNorm"
    SEQ = "Sequential"
    RES = "Residual"


# =============================================================================
#  Internal helpers
# =============================================================================


def _make_activation(kind: ActivationKind, width: int) -> nn.Module:
    return zReLU() if kind is ActivationKind.Z_RELU else modReLU(width)


def _seq(*mods: nn.Module) -> nn.Module:
    return mods[0] if len(mods) == 1 else ComplexSequential(*mods)


def _maybe_activate(
    module: nn.Module, act: Optional["ActivationCfg"], width: int
) -> nn.Module:
    return _seq(module, _make_activation(act.kind, width)) if act else module


def _maybe_project(module: nn.Module, in_w: int, out_w: int) -> Tuple[nn.Module, int]:
    return (
        (module, in_w)
        if in_w == out_w
        else (_seq(module, ComplexLinear(in_w, out_w)), out_w)
    )


@contextmanager
def _default_dtype(dtype: torch.dtype) -> Iterator[None]:
    prev = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(prev)


@contextmanager
def _default_device(device: torch.device) -> Iterator[None]:
    """Temporarily route *all* new tensors to *device* (if API available)."""
    if hasattr(torch, "set_default_device"):
        prev_dev = torch.tensor([], device="cpu").device  # current default
        torch.set_default_device(str(device))
        try:
            yield
        finally:
            torch.set_default_device(str(prev_dev))
    else:  # older PyTorch – build on CPU, move once at the end
        yield


# =============================================================================
#  Pydantic configs – pure data
# =============================================================================


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
    "LinearCfg", "NaiveBNCfg", "CovBNCfg", "SequentialCfg", "ResidualCfg"
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
    """Network topology – pure data, no behaviour."""

    layers: List[LayerCfg]
    seed: PositiveInt
    final_activation: Optional[ActivationCfg] = None

    def __eq__(self, other: object) -> bool:  # noqa: D401
        return isinstance(other, CVNNConfig) and self.model_dump() == other.model_dump()


# =============================================================================
#  Recursive builder
# =============================================================================


def _build_from_cfg(cfg: LayerCfg, cur_w: int) -> Tuple[nn.Module, int]:
    """Recursively convert *cfg* into an :class:`nn.Module`."""
    match cfg:
        # ── ComplexLinear ────────────────────────────────────────────────
        case LinearCfg() as c:
            out_w = c.width or cur_w
            layer = ComplexLinear(cur_w, out_w, bias=c.bias)
            return _maybe_activate(layer, c.activation, out_w), out_w

        # ── Naive BN ─────────────────────────────────────────────────────
        case NaiveBNCfg() as c:
            nbn = NaiveComplexBatchNorm(
                cur_w,
                eps=c.eps,
                momentum=c.momentum,
                affine=c.affine,
                track_running_stats=c.track_running_stats,
            )
            return _maybe_activate(nbn, c.activation, cur_w), cur_w

        # ── Covariance BN ───────────────────────────────────────────────
        case CovBNCfg() as c:
            cbn = CovarianceComplexBatchNorm(
                cur_w,
                eps=c.eps,
                momentum=c.momentum,
                affine=c.affine,
                track_running_stats=c.track_running_stats,
            )
            return _maybe_activate(cbn, c.activation, cur_w), cur_w

        # ── Sequential container ────────────────────────────────────────
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

        # ── Residual block ──────────────────────────────────────────────
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
    """Materialise a CVNN described by *cfg* on the requested device / dtype."""
    torch.manual_seed(cfg.seed)

    tgt_device = torch.device(device) if device is not None else torch.device("cpu")
    tgt_dtype = dtype or torch.get_default_dtype()

    # Build under controlled default dtype & device so tensors spawn correctly.
    with _default_dtype(tgt_dtype), _default_device(tgt_device):
        body, w = _build_from_cfg(SequentialCfg(layers=cfg.layers), n_inputs)
        body, w = _maybe_project(body, w, n_outputs)
        net = _maybe_activate(body, cfg.final_activation, w)

    # Fallback copy if set_default_device was unavailable.
    if any(p.device != tgt_device for p in net.parameters()):
        net = net.to(device=tgt_device, dtype=tgt_dtype)

    return net
