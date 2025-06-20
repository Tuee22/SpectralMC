"""cvnn_factory.py
=================
A *purely-functional* **factory** for building complex-valued neural networks
(CVNNs) from a nested :class:`CVNNConfig` description.

The public surface area is intentionally small – :func:`build_model` is the
single entry-point.  Every layer, residual branch, and activation is created in
an *expression-oriented* manner (i.e. no mutable loops).  The result is an
initialised :class:`torch.nn.Module` built **directly on the requested
``device``** (CPU, CUDA, …) by wrapping model construction in a default-device
context rather than moving modules afterwards.

All symbols in this file type-check under **mypy --strict** with the standard
PyTorch stubs.
"""

from __future__ import annotations

from contextlib import nullcontext
from enum import Enum
from typing import List, Optional, Tuple, TypeAlias, Union

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
    """Return a freshly-constructed activation module for *kind*."""
    return zReLU() if kind is ActivationKind.Z_RELU else modReLU(width)


def _seq(*mods: nn.Module) -> nn.Module:  # noqa: D401
    """Return *mods* unchanged or wrapped in :class:`ComplexSequential`.

    Single modules are returned verbatim to avoid an extra call-frame in the
    forward pass.
    """
    return mods[0] if len(mods) == 1 else ComplexSequential(*mods)


def _maybe_activate(
    module: nn.Module,
    act: Optional["ActivationCfg"],
    width: int,
) -> nn.Module:
    """Append an activation to *module* when *act* is not *None*."""
    return _seq(module, _make_activation(act.kind, width)) if act else module


def _maybe_project(
    module: nn.Module,
    in_w: int,
    out_w: int,
) -> Tuple[nn.Module, int]:
    """If *in_w* != *out_w* append a :class:`ComplexLinear` projection."""
    if in_w == out_w:
        return module, in_w
    return _seq(module, ComplexLinear(in_w, out_w)), out_w


# =============================================================================
#  Pydantic configs – these are *pure data* (no behaviour!)
# =============================================================================


class ActivationCfg(BaseModel):
    """Config for a complex activation layer."""

    kind: ActivationKind


class LinearCfg(BaseModel):
    """Config for :class:`~spectralmc.cvnn.ComplexLinear`."""

    kind: LayerKind = LayerKind.LINEAR
    width: Optional[int] = None  # Defaults to *input width* if ``None``
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


# A recursive type alias covering *all* layer configuration variants.
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
    body: SequentialCfg  # main path
    projection: Optional[LinearCfg] = None
    activation: Optional[ActivationCfg] = None


class CVNNConfig(BaseModel):
    """Top-level network specification consumed by :func:`build_model`."""

    layers: List[LayerCfg]
    seed: PositiveInt  # seeding for reproducible weight init
    final_activation: Optional[ActivationCfg] = None


# =============================================================================
#  Recursive builder (private)
# =============================================================================


def _build_from_cfg(
    cfg: LayerCfg,
    cur_w: int,
) -> Tuple[nn.Module, int]:
    """Recursively translate *cfg* to an :class:`nn.Module`.

    Parameters
    ----------
    cfg
        The configuration node to materialise.
    cur_w
        The current feature width flowing *into* this node.  Some layers use the
        value to infer missing hyper-parameters (e.g. a *Linear* layer whose
        ``width`` is not specified).

    Returns
    -------
    Tuple[nn.Module, int]
        The freshly-built sub-module **and** the feature width leaving this
        sub-module.  The latter feeds into the next element when constructing a
        :class:`SequentialCfg`.
    """

    match cfg:
        # ── Linear ────────────────────────────────────────────────────────
        case LinearCfg() as c:
            out_w: int = c.width or cur_w
            lin: nn.Module = ComplexLinear(cur_w, out_w, bias=c.bias)
            return _maybe_activate(lin, c.activation, out_w), out_w

        # ── Naive BN ──────────────────────────────────────────────────────
        case NaiveBNCfg() as c:
            bn_naive: nn.Module = NaiveComplexBatchNorm(
                num_features=cur_w,
                eps=c.eps,
                momentum=c.momentum,
                affine=c.affine,
                track_running_stats=c.track_running_stats,
            )
            return _maybe_activate(bn_naive, c.activation, cur_w), cur_w

        # ── Covariance BN ────────────────────────────────────────────────
        case CovBNCfg() as c:
            bn_cov: nn.Module = CovarianceComplexBatchNorm(
                num_features=cur_w,
                eps=c.eps,
                momentum=c.momentum,
                affine=c.affine,
                track_running_stats=c.track_running_stats,
            )
            return _maybe_activate(bn_cov, c.activation, cur_w), cur_w

        # ── Sequential ───────────────────────────────────────────────────
        case SequentialCfg() as c:

            def _fold(lst: List[LayerCfg], w_in: int) -> Tuple[List[nn.Module], int]:
                """Pure *right-fold* over *lst* building modules on the fly."""
                if not lst:
                    return ([], w_in)
                head, *tail = lst
                head_mod, w_mid = _build_from_cfg(head, w_in)
                tail_mods, w_out = _fold(tail, w_mid)
                return ([head_mod, *tail_mods], w_out)

            mods, width = _fold(c.layers, cur_w)
            seq_mod: nn.Module = _seq(*mods)
            return _maybe_activate(seq_mod, c.activation, width), width

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

            # Sanity check – should never fail if config is well-formed.
            assert proj_w == body_w, "Projection width mismatch in Residual block"

            res_mod: nn.Module = ComplexResidual(
                body=body_mod,
                proj=proj_mod,  # Optional[nn.Module]
                post_act=(
                    _make_activation(c.activation.kind, body_w)
                    if c.activation
                    else None
                ),
            )
            return res_mod, body_w

    # Defensive fallback – *mypy* knows we are exhaustive but Python does not.
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
) -> nn.Module:
    """Materialise a CVNN described by *cfg*.

    Parameters
    ----------
    n_inputs
        Width of the real/imaginary input vectors.
    n_outputs
        Desired width of the network output.
    cfg
        A fully-specified :class:`CVNNConfig` instance.
    device
        Optional target device (**CPU**, **CUDA**, *etc.*).  When provided, the
        entire model is constructed inside a ``with torch.device(device):`` block
        so that **all parameters are born on that device**.  Pass ``None`` to
        build on the default CPU.

    Returns
    -------
    torch.nn.Module
        The initialised model (already residing on *device* when specified).
    """

    # Reproducible initialisation.
    torch.manual_seed(int(cfg.seed))

    # Helper to build the network body (invoked inside/outside the context).
    def _materialise() -> nn.Module:
        body, w = _build_from_cfg(SequentialCfg(layers=cfg.layers), n_inputs)
        body, w = _maybe_project(body, w, n_outputs)
        return _maybe_activate(body, cfg.final_activation, w)

    if device is None:
        return _materialise()

    # ``torch.device`` is itself a context manager; convert non-str input.
    dev_str: str = device if isinstance(device, str) else str(device)
    # mypy stubs for torch lack context-manager methods on ``device``, so we
    # silence the attribute error check with ``type: ignore``.
    with torch.device(dev_str):
        return _materialise()
