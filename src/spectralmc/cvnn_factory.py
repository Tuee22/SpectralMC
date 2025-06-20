# cvnn_factory_generalized.py
"""Build **flexible, closed** complex-valued neural networks from a Pydantic
configuration.

Highlights
----------
* Expression‑oriented, purely functional construction (no mutable loops).
* Helper combinators `_maybe_activate` / `_maybe_project` remove repetition.
* mypy‑strict clean; compatible with `black`.
"""

from __future__ import annotations

from enum import Enum
from typing import List, Optional, Tuple, TypeAlias, Union

import torch
import torch.nn as nn
from pydantic import BaseModel, Field, PositiveInt

from spectralmc.cvnn import (
    ComplexLinear,
    ComplexResidual,
    ComplexSequential,
    CovarianceComplexBatchNorm,
    NaiveComplexBatchNorm,
    modReLU,
    zReLU,
)

# =============================================================================
#  Enumerations
# =============================================================================


class ActivationKind(str, Enum):
    Z_RELU = "zReLU"
    MOD_RELU = "modReLU"


class LayerKind(str, Enum):
    LINEAR = "ComplexLinear"
    BN_NAIVE = "NaiveComplexBatchNorm"
    BN_COV = "CovarianceComplexBatchNorm"
    SEQ = "Sequential"
    RES = "Residual"


# =============================================================================
#  Helper combinators
# =============================================================================


def _make_activation(kind: "ActivationKind", width: int) -> nn.Module:
    return zReLU() if kind is ActivationKind.Z_RELU else modReLU(width)


def _seq(*mods: nn.Module) -> nn.Module:
    """Return the lone module unchanged or wrap many in `ComplexSequential`."""
    return mods[0] if len(mods) == 1 else ComplexSequential(*mods)


def _maybe_activate(
    module: nn.Module, act: Optional["ActivationCfg"], width: int
) -> nn.Module:
    return _seq(module, _make_activation(act.kind, width)) if act else module


def _maybe_project(module: nn.Module, in_w: int, out_w: int) -> Tuple[nn.Module, int]:
    return (
        (_seq(module, ComplexLinear(in_w, out_w)), out_w)
        if in_w != out_w
        else (module, in_w)
    )


# =============================================================================
#  Pydantic configs
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
    "LinearCfg",
    "NaiveBNCfg",
    "CovBNCfg",
    "SequentialCfg",
    "ResidualCfg",
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
    layers: List[LayerCfg]
    seed: PositiveInt
    final_activation: Optional[ActivationCfg] = None


# =============================================================================
#  Recursive builder
# =============================================================================


def _build_from_cfg(cfg: LayerCfg, cur_w: int) -> Tuple[nn.Module, int]:
    match cfg:
        # ─── Linear ────────────────────────────────────────────────────────
        case LinearCfg() as c:
            out_w = c.width or cur_w
            lin: nn.Module = ComplexLinear(cur_w, out_w, bias=c.bias)
            return _maybe_activate(lin, c.activation, out_w), out_w

        # ─── Naive BN ──────────────────────────────────────────────────────
        case NaiveBNCfg() as c:
            bn_naive: nn.Module = NaiveComplexBatchNorm(
                num_features=cur_w,
                eps=c.eps,
                momentum=c.momentum,
                affine=c.affine,
                track_running_stats=c.track_running_stats,
            )
            return _maybe_activate(bn_naive, c.activation, cur_w), cur_w

        # ─── Covariance BN ────────────────────────────────────────────────
        case CovBNCfg() as c:
            bn_cov: nn.Module = CovarianceComplexBatchNorm(
                num_features=cur_w,
                eps=c.eps,
                momentum=c.momentum,
                affine=c.affine,
                track_running_stats=c.track_running_stats,
            )
            return _maybe_activate(bn_cov, c.activation, cur_w), cur_w

        # ─── Sequential ───────────────────────────────────────────────────
        case SequentialCfg() as c:

            def _fold(lst: List[LayerCfg], w_in: int) -> Tuple[List[nn.Module], int]:
                if not lst:
                    return ([], w_in)

                h, *t = lst
                h_mod, w_mid = _build_from_cfg(h, w_in)
                t_mods, w_out = _fold(t, w_mid)
                return ([h_mod, *t_mods], w_out)

            mods, width = _fold(c.layers, cur_w)
            seq_mod = _seq(*mods)
            return _maybe_activate(seq_mod, c.activation, width), width

            # ─── Residual ─────────────────────────────────────────────────────
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

            # Guard against accidental mismatch without an explicit `if` chain.
            assert proj_w == body_w, "Projection width mismatch in Residual block"

            res_mod = ComplexResidual(
                body=body_mod,
                proj=proj_mod,
                post_act=(
                    _make_activation(c.activation.kind, body_w)
                    if c.activation
                    else None
                ),
            )
            return res_mod, body_w

    # Defensive fallback
    raise RuntimeError(f"Unhandled layer config type: {type(cfg).__name__}")


# =============================================================================
#  Public factory
# =============================================================================


def build_model(*, n_inputs: int, n_outputs: int, cfg: CVNNConfig) -> nn.Module:
    torch.manual_seed(cfg.seed)

    initial_body, initial_w = _build_from_cfg(
        SequentialCfg(layers=cfg.layers), n_inputs
    )
    projected_body, final_w = _maybe_project(initial_body, initial_w, n_outputs)
    final_body = _maybe_activate(projected_body, cfg.final_activation, final_w)
    return final_body


__all__: Tuple[str, ...] = ("CVNNConfig", "build_model")
