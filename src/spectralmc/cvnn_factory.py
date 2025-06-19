# cvnn_factory_generalized.py
"""Build **flexible, closed** complex-valued neural networks from a Pydantic
configuration.

Key ideas
---------
* **Input / output agnostic** – any ``n_inputs``/``n_outputs`` pair works.  The
  builder inserts complex-linear projections wherever widths disagree.
* **Width inference** – a ``ComplexLinear`` whose ``width`` is ``None``
  propagates the incoming width unchanged.
* **Optional global activation** – a final activation can be specified once in
  the root config.
* **Functional flavour** – helper utilities avoid mutable state and rely on
  pure functions, recursion, comprehensions, and pattern matching.
* **mypy-strict clean** – no ``Any`` and no ``# type: ignore``.
"""

from __future__ import annotations

import functools
from enum import Enum
from itertools import chain
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
    """Supported complex activation layers."""

    Z_RELU = "zReLU"
    MOD_RELU = "modReLU"


class LayerKind(str, Enum):
    """Atomic and composite layer kinds supported by the builder."""

    LINEAR = "ComplexLinear"
    BN_NAIVE = "NaiveComplexBatchNorm"
    BN_COV = "CovarianceComplexBatchNorm"
    SEQ = "Sequential"
    RES = "Residual"


# =============================================================================
#  Pure helper functions
# =============================================================================


def _make_activation(kind: ActivationKind, width: int) -> nn.Module:
    """Return the torch implementation of *kind*."""

    return zReLU() if kind is ActivationKind.Z_RELU else modReLU(width)


def _flatten_module(module: nn.Module) -> List[nn.Module]:
    """Recursively collect atomic layers, eliminating nested sequentials."""

    if isinstance(module, ComplexSequential):
        return list(chain.from_iterable(_flatten_module(m) for m in module.layers))
    return [module]


def _seq(*modules: nn.Module) -> nn.Module:
    """Compose modules into a single (flattened) :class:`ComplexSequential`."""

    flat_layers: List[nn.Module] = list(
        chain.from_iterable(_flatten_module(m) for m in modules)
    )
    return flat_layers[0] if len(flat_layers) == 1 else ComplexSequential(*flat_layers)


# =============================================================================
#  Pydantic layer descriptions
# =============================================================================


class ActivationCfg(BaseModel):
    kind: ActivationKind


class LinearCfg(BaseModel):
    kind: LayerKind = LayerKind.LINEAR
    width: Optional[int] = None  # None ⇒ keep current width
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
    seed: PositiveInt = Field(..., description="RNG seed for deterministic init")
    final_activation: Optional[ActivationCfg] = None


# =============================================================================
#  Width-aware recursive builder
# =============================================================================


def _build_from_cfg(cfg: LayerCfg, cur_w: int) -> Tuple[nn.Module, int]:
    """Convert *cfg* into a torch module and report its output width."""

    match cfg:
        # ─── Linear ────────────────────────────────────────────────────────
        case LinearCfg() as lin_cfg:
            out_w = lin_cfg.width if lin_cfg.width is not None else cur_w
            layer: nn.Module = ComplexLinear(cur_w, out_w, bias=lin_cfg.bias)
            return (
                (
                    _seq(layer, _make_activation(lin_cfg.activation.kind, out_w))
                    if lin_cfg.activation
                    else layer
                ),
                out_w,
            )

        # ─── Naive BN ──────────────────────────────────────────────────────
        case NaiveBNCfg() as bn_cfg:
            layer = NaiveComplexBatchNorm(
                num_features=cur_w,
                eps=bn_cfg.eps,
                momentum=bn_cfg.momentum,
                affine=bn_cfg.affine,
                track_running_stats=bn_cfg.track_running_stats,
            )
            return (
                (
                    _seq(layer, _make_activation(bn_cfg.activation.kind, cur_w))
                    if bn_cfg.activation
                    else layer
                ),
                cur_w,
            )

        # ─── Covariance BN ────────────────────────────────────────────────
        case CovBNCfg() as cbn_cfg:
            layer = CovarianceComplexBatchNorm(
                num_features=cur_w,
                eps=cbn_cfg.eps,
                momentum=cbn_cfg.momentum,
                affine=cbn_cfg.affine,
                track_running_stats=cbn_cfg.track_running_stats,
            )
            return (
                (
                    _seq(layer, _make_activation(cbn_cfg.activation.kind, cur_w))
                    if cbn_cfg.activation
                    else layer
                ),
                cur_w,
            )

        # ─── Sequential composite ─────────────────────────────────────────
        case SequentialCfg() as seq_cfg:
            State: TypeAlias = Tuple[List[nn.Module], int]

            def _accumulate(state: State, sub_cfg: LayerCfg) -> State:
                built, width_in = state
                sub_mod, width_out = _build_from_cfg(sub_cfg, width_in)
                return (built + [sub_mod], width_out)

            init_state: State = ([], cur_w)
            modules_list, width = functools.reduce(
                _accumulate,
                seq_cfg.layers,
                init_state,
            )
            seq_layer = _seq(*modules_list)
            return (
                (
                    _seq(seq_layer, _make_activation(seq_cfg.activation.kind, width))
                    if seq_cfg.activation
                    else seq_layer
                ),
                width,
            )

        # ─── Residual composite ───────────────────────────────────────────
        case ResidualCfg() as res_cfg:
            body_mod, body_w = _build_from_cfg(res_cfg.body, cur_w)

            # Decide projection for skip path.
            if res_cfg.projection is not None:
                proj_mod, proj_w = _build_from_cfg(res_cfg.projection, cur_w)
                if proj_w != body_w:
                    raise ValueError(
                        f"Projection width {proj_w} must equal body width {body_w}",
                    )
            elif body_w != cur_w:
                proj_mod = ComplexLinear(cur_w, body_w)
            else:
                proj_mod = None

            residual_layer = ComplexResidual(
                body=body_mod,
                proj=proj_mod,
                post_act=(
                    _make_activation(res_cfg.activation.kind, body_w)
                    if res_cfg.activation
                    else None
                ),
            )
            return residual_layer, body_w

    # ─── Defensive fallback ───────────────────────────────────────────────
    raise RuntimeError(f"Unhandled layer config type: {type(cfg).__name__}")


# =============================================================================
#  Public factory
# =============================================================================


def build_model(*, n_inputs: int, n_outputs: int, cfg: CVNNConfig) -> nn.Module:
    """Instantiate a CVNN matching *cfg* for arbitrary input/output widths."""

    torch.manual_seed(cfg.seed)

    # Treat top-level list as an implicit Sequential.
    top_seq = SequentialCfg(layers=cfg.layers)
    body, body_w = _build_from_cfg(top_seq, n_inputs)

    # Final width projection if needed.
    if body_w != n_outputs:
        body = _seq(body, ComplexLinear(body_w, n_outputs))
        body_w = n_outputs

    # Optional global activation.
    if cfg.final_activation is not None:
        body = _seq(body, _make_activation(cfg.final_activation.kind, body_w))

    return body


__all__: Tuple[str, ...] = ("CVNNConfig", "build_model")
