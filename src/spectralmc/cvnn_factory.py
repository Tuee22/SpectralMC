# src/spectralmc/cvnn_factory.py
"""
Build complex-valued neural networks from a Pydantic description.

Key points
----------
*  The JSON/YAML config encodes only **topology**.  The overall input and
   output widths (`n_inputs`, `n_outputs`) are passed to
   :func:`build_model`.
*  Supports arbitrary nesting with two composite layers:
   `Sequential` and `Residual`.
*  Performs runtime width checks and, where necessary, inserts a
   projection inside residual blocks so the addition is valid.
*  Contains **no** checkpoint/weight-loading logic – this module’s sole
   job is to create *blank* networks.
*  Fully typed **without** ``Any`` or ``# type: ignore``; passes
   ``mypy --strict`` when `types-torch` stubs are available.
"""

from __future__ import annotations

from typing import List, Literal, Optional, Tuple, TypeAlias, Union

import torch
import torch.nn as nn
from pydantic import BaseModel, Field

# ────────────────────────────────────────────────────────────────
#  Core layers (defined in spectralmc.cvnn)
# ────────────────────────────────────────────────────────────────
from spectralmc.cvnn import (
    ComplexLinear,
    CovarianceComplexBatchNorm,
    NaiveComplexBatchNorm,
    modReLU,
    zReLU,
    ComplexSequential,
    ComplexResidual,
)


# =============================================================================
# Pydantic layer descriptions
# =============================================================================
class ActivationCfg(BaseModel):
    """Descriptor for an activation layer."""

    kind: Literal["zReLU", "modReLU"]


def _make_activation(kind: str, width: int) -> nn.Module:
    """Instantiate an activation of *kind* for the given *width*."""
    return zReLU() if kind == "zReLU" else modReLU(width)


class LinearCfg(BaseModel):
    """Complex fully-connected layer."""

    kind: Literal["ComplexLinear"] = "ComplexLinear"
    width: Optional[int] = None
    bias: bool = True
    activation: Optional[ActivationCfg] = None


class NaiveBNCfg(BaseModel):
    """BatchNorm that treats real & imag parts independently."""

    kind: Literal["NaiveComplexBatchNorm"] = "NaiveComplexBatchNorm"
    eps: float = 1e-5
    momentum: float = 0.1
    affine: bool = True
    track_running_stats: bool = True
    activation: Optional[ActivationCfg] = None


class CovBNCfg(BaseModel):
    """Covariance-whitening BatchNorm (Trabelsi et al., 2018)."""

    kind: Literal["CovarianceComplexBatchNorm"] = "CovarianceComplexBatchNorm"
    eps: float = 1e-5
    momentum: float = 0.1
    affine: bool = True
    track_running_stats: bool = True
    activation: Optional[ActivationCfg] = None


class SequentialCfg(BaseModel):  # forward-ref uses quotes below
    """A list of layers executed sequentially."""

    kind: Literal["Sequential"] = "Sequential"
    layers: List[LayerCfg]
    activation: Optional[ActivationCfg] = None


class ResidualCfg(BaseModel):
    """Residual wrapper around a nested *body* sequence."""

    kind: Literal["Residual"] = "Residual"
    body: SequentialCfg
    projection: Optional[LinearCfg] = None
    activation: Optional[ActivationCfg] = None


LayerCfg: TypeAlias = Union[
    LinearCfg,
    NaiveBNCfg,
    CovBNCfg,
    SequentialCfg,
    ResidualCfg,
]


class CVNNConfig(BaseModel):
    """Root model that serialises a complete network topology."""

    layers: List[LayerCfg]
    seed: int = Field(gt=0)


# =============================================================================
# Recursive builder
# =============================================================================
def _build_from_cfg(cfg: LayerCfg, cur_w: int) -> Tuple[nn.Module, int]:
    """
    Recursively convert *cfg* into an :class:`nn.Module`.

    Args:
        cfg:   The layer (or composite) to build.
        cur_w: Current feature width entering this node.

    Returns:
        tuple(module, out_width)
    """
    # ───── ComplexLinear ────────────────────────────────────────────
    if isinstance(cfg, LinearCfg):
        out_w = cfg.width or cur_w
        module: nn.Module = ComplexLinear(cur_w, out_w, bias=cfg.bias)
        if cfg.activation is not None:
            module = ComplexSequential(
                module, _make_activation(cfg.activation.kind, out_w)
            )
        return module, out_w

    # ───── Naive BN ────────────────────────────────────────────────
    if isinstance(cfg, NaiveBNCfg):
        module = NaiveComplexBatchNorm(
            num_features=cur_w,
            eps=cfg.eps,
            momentum=cfg.momentum,
            affine=cfg.affine,
            track_running_stats=cfg.track_running_stats,
        )
        if cfg.activation is not None:
            module = ComplexSequential(
                module, _make_activation(cfg.activation.kind, cur_w)
            )
        return module, cur_w

    # ───── Covariance BN ───────────────────────────────────────────
    if isinstance(cfg, CovBNCfg):
        module = CovarianceComplexBatchNorm(
            num_features=cur_w,
            eps=cfg.eps,
            momentum=cfg.momentum,
            affine=cfg.affine,
            track_running_stats=cfg.track_running_stats,
        )
        if cfg.activation is not None:
            module = ComplexSequential(
                module, _make_activation(cfg.activation.kind, cur_w)
            )
        return module, cur_w

    # ───── Sequential composite ────────────────────────────────────
    if isinstance(cfg, SequentialCfg):
        modules: List[nn.Module] = []
        width = cur_w
        for sub in cfg.layers:
            sub_mod, width = _build_from_cfg(sub, width)
            modules.append(sub_mod)
        seq: nn.Module = ComplexSequential(*modules)
        if cfg.activation is not None:
            seq = ComplexSequential(seq, _make_activation(cfg.activation.kind, width))
        return seq, width

    # ───── Residual composite ──────────────────────────────────────
    if isinstance(cfg, ResidualCfg):
        body_mod, body_w = _build_from_cfg(cfg.body, cur_w)

        if cfg.projection is not None:
            proj_mod, proj_w = _build_from_cfg(cfg.projection, cur_w)
            if proj_w != body_w:
                raise ValueError(
                    f"Projection width {proj_w} must equal body width {body_w}.",
                )
        elif body_w != cur_w:
            proj_mod = ComplexLinear(cur_w, body_w)
        else:
            proj_mod = None

        res_mod = ComplexResidual(
            body=body_mod,
            proj=proj_mod,
            post_act=(
                _make_activation(cfg.activation.kind, body_w)
                if cfg.activation is not None
                else None
            ),
        )
        return res_mod, body_w

    # ───── unreachable (defensive) ─────────────────────────────────
    raise RuntimeError(f"Unhandled layer config: {type(cfg).__name__}")


# =============================================================================
# Public factory
# =============================================================================
def build_model(*, n_inputs: int, n_outputs: int, cfg: CVNNConfig) -> nn.Module:
    """
    Construct a complex-valued network from *cfg*.

    Args:
        n_inputs:  Feature width of the real (and imag) input tensors.
        n_outputs: Desired feature width of the final output tensors.
        cfg:       A validated :class:`CVNNConfig`.

    Returns:
        A freshly initialised :class:`nn.Module`.
    """

    torch.manual_seed(cfg.seed)

    # Treat the top level as an implicit Sequential.
    top = SequentialCfg(layers=cfg.layers)
    body, body_w = _build_from_cfg(top, n_inputs)

    # Append a final linear if widths differ.
    if body_w != n_outputs:
        body = ComplexSequential(body, ComplexLinear(body_w, n_outputs))

    return body


__all__ = ["CVNNConfig", "build_model"]
