"""
Strict, project‑specific stub for the **top‑level** :pymod:`torch` namespace.

The goal is *just enough* typing so that **mypy ``--strict``** passes for the
SpectralMC code‑base and test‑suite – **nothing more**.  All public symbols we
use are declared with precise signatures; everything else is intentionally
absent so that accidental dependencies are caught early.

Conventions
-----------
* No ``Any``, ``cast`` or ``type: ignore`` markers – the stub must remain
  type‑pure.
* Each group of helpers and sub‑modules is collected in one place so the file
  stays maintainable.
"""

from __future__ import annotations

import builtins as _b
from typing import Iterator, Sequence, Tuple, TypeAlias, TypeVar, overload

# --------------------------------------------------------------------------- #
#  dtype & device singletons – minimal surface                                #
# --------------------------------------------------------------------------- #
class dtype: ...

float32: dtype
float64: dtype
float16: dtype
bfloat16: dtype
complex64: dtype
complex128: dtype
int64: dtype
float: dtype  # alias maintained by PyTorch
double: dtype  # ditto
long: dtype  # ditto

def get_default_dtype() -> dtype: ...
def set_default_dtype(d: dtype) -> None: ...

class device:
    """Very small subset of :class:`torch.device` (still a context manager)."""

    def __init__(self, spec: str | "device" | None = ...) -> None: ...
    def __enter__(self) -> "device": ...
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object | None,
    ) -> bool | None: ...
    # PyTorch exposes ``cpu`` / ``cuda`` / ``mps`` etc.
    @property
    def type(self) -> str: ...
    @property
    def index(self) -> int | None: ...

_TorchDevice: TypeAlias = device
_DType = dtype  # shorthand used throughout the stub

# --------------------------------------------------------------------------- #
#  Tensor – *very* trimmed, but covers every method SpectralMC touches        #
# --------------------------------------------------------------------------- #
_TTensor = TypeVar("_TTensor", bound="Tensor")

class Tensor:
    """Reduced ``torch.Tensor`` façade.

    Only members actually referenced by production / test code are declared.
    """

    # ───── construction & autograd ───────────────────────────────────────
    def __init__(
        self,
        *size: int,
        dtype: _DType | None = ...,
        requires_grad: bool = ...,
    ) -> None: ...

    grad: "Tensor | None"
    requires_grad: bool

    def backward(self, gradient: "Tensor | None" = ...) -> None: ...

    # ───── core properties ──────────────────────────────────────────────
    @property
    def dtype(self) -> _DType: ...
    @property
    def shape(self) -> tuple[int, ...]: ...
    @property
    def ndim(self) -> int: ...
    @property
    def device(self) -> _TorchDevice: ...
    @property
    def T(self) -> "Tensor": ...

    # Presence checks used by the test‑suite
    # --------------------------------------
    def is_floating_point(self) -> bool: ...
    @property
    def is_cuda(self) -> bool: ...

    # ───── simple math helpers ──────────────────────────────────────────
    def mean(self, dim: int | None = ..., **kw: object) -> "Tensor": ...
    def var(
        self,
        dim: int | None = ...,
        *,
        unbiased: bool = ...,
        **kw: object,
    ) -> "Tensor": ...
    def pow(self, exponent: _b.int | _b.float) -> "Tensor": ...
    def square(self) -> "Tensor": ...
    def abs(self) -> "Tensor": ...
    def all(self) -> "Tensor": ...

    # ───── reductions & transformations ─────────────────────────────────
    def max(self, dim: int | None = ..., keepdim: bool = ...) -> "Tensor": ...
    def sum(
        self,
        dim: int | None = ...,
        keepdim: bool = ...,
        *,
        dtype: _DType | None = ...,
    ) -> "Tensor": ...
    def unsqueeze(self, dim: int) -> "Tensor": ...
    def squeeze(self, dim: int | None = ...) -> "Tensor": ...
    def transpose(self, dim0: int, dim1: int) -> "Tensor": ...

    # ───── arithmetic operators (binary *and* in‑place) ─────────────────
    def __add__(self, other: "Tensor | _b.float | _b.int") -> "Tensor": ...
    __radd__ = __add__
    def __sub__(self, other: "Tensor | _b.float | _b.int") -> "Tensor": ...
    __rsub__ = __sub__
    def __mul__(self, other: "Tensor | _b.float | _b.int") -> "Tensor": ...
    __rmul__ = __mul__
    def __truediv__(self, other: "Tensor | _b.float | _b.int") -> "Tensor": ...
    def __rtruediv__(self, other: "Tensor | _b.float | _b.int") -> "Tensor": ...
    def __matmul__(self, other: "Tensor") -> "Tensor": ...

    # In‑place variants explicitly referenced in code/tests
    def add_(self, other: "Tensor | _b.float | _b.int") -> "Tensor": ...
    def mul_(self, other: "Tensor | _b.float | _b.int") -> "Tensor": ...
    def sub_(self, other: "Tensor | _b.float | _b.int") -> "Tensor": ...
    def copy_(self, other: "Tensor") -> "Tensor": ...
    def zero_(self) -> "Tensor": ...
    def fill_(self, value: _b.int | _b.float) -> "Tensor": ...

    # ───── comparisons / boolean ops ────────────────────────────────────
    def __lt__(self, other: "Tensor | _b.float | _b.int") -> "Tensor": ...
    def __le__(self, other: "Tensor | _b.float | _b.int") -> "Tensor": ...
    def __gt__(self, other: "Tensor | _b.float | _b.int") -> "Tensor": ...
    def __ge__(self, other: "Tensor | _b.float | _b.int") -> "Tensor": ...
    def __and__(self, other: "Tensor | bool") -> "Tensor": ...
    __rand__ = __and__

    def __getitem__(self, idx: object) -> "Tensor": ...

    # ───── utilities frequently used in tests ───────────────────────────
    def detach(self) -> "Tensor": ...
    def cpu(self) -> "Tensor": ...
    def clone(self) -> "Tensor": ...
    def reshape(
        self,
        shape: Tuple[int, ...] | Sequence[int] | int,
        *more: int,
    ) -> "Tensor": ...
    def tolist(self) -> list[_b.int | _b.float]: ...
    def item(self) -> _b.int | _b.float: ...
    def to(
        self: _TTensor,
        dtype: _DType | None = ...,
        device: _TorchDevice | str | None = ...,
        copy: bool | None = ...,
        non_blocking: bool | None = ...,
    ) -> _TTensor: ...
    def equal(self, other: "Tensor") -> bool: ...
    def __iter__(self) -> Iterator["Tensor"]: ...

# --------------------------------------------------------------------------- #
#  Top‑level functional helpers                                               #
# --------------------------------------------------------------------------- #
def is_floating_point(t: Tensor, /) -> bool: ...
def zeros(*size: int, dtype: _DType | None = ...) -> Tensor: ...
def ones(
    *size: int,
    dtype: _DType | None = ...,
    device: _TorchDevice | str | None = ...,
) -> Tensor: ...
def full(
    size: Tuple[int, ...],
    fill_value: _b.int | _b.float,
    dtype: _DType | None = ...,
) -> Tensor: ...
def full_like(
    a: Tensor,
    fill_value: _b.int | _b.float,
    *,
    dtype: _DType | None = ...,
    device: _TorchDevice | str | None = ...,
) -> Tensor: ...
def zeros_like(a: Tensor) -> Tensor: ...
def randn(
    *size: int,
    dtype: _DType | None = ...,
    device: _TorchDevice | str | None = ...,
    requires_grad: bool = ...,
) -> Tensor: ...
def tensor(
    data: object,
    *,
    dtype: _DType | None = ...,
    device: _TorchDevice | str | None = ...,
    requires_grad: bool = ...,
) -> Tensor: ...
def matmul(a: Tensor, b: Tensor) -> Tensor: ...
def sqrt(a: Tensor) -> Tensor: ...
def clamp(
    a: Tensor,
    *,
    min: _b.float | None = ...,
    max: _b.float | None = ...,
) -> Tensor: ...
def relu(a: Tensor) -> Tensor: ...
def stack(tensors: Sequence[Tensor], dim: int = ...) -> Tensor: ...
def square(a: Tensor) -> Tensor: ...
def all(a: Tensor) -> bool: ...
def allclose(
    a: Tensor,
    b: Tensor,
    *,
    atol: _b.float = ...,
    rtol: _b.float = ...,
) -> bool: ...
def isfinite(a: Tensor) -> Tensor: ...
def manual_seed(seed: int) -> None: ...

# mathematics helpers not exposed as Tensor methods in stub
abs = Tensor.abs  # re‑export so ``torch.abs`` is recognised

# --------------------------------------------------------------------------- #
#  Autograd guard                                                             #
# --------------------------------------------------------------------------- #
class _NoGrad:
    def __enter__(self) -> None: ...
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object | None,
    ) -> None: ...

def no_grad() -> _NoGrad: ...

# --------------------------------------------------------------------------- #
#  Structured sub‑modules – we stub only what we import                       #
# --------------------------------------------------------------------------- #
from . import nn as nn  # noqa: F401
from . import optim as optim  # noqa: F401
from . import cuda as cuda  # noqa: F401
from . import utils as utils  # noqa: F401
from . import fft as fft  # runtime sub‑module
from . import linalg as linalg
from . import random as random  # noqa: F401

# --------------------------------------------------------------------------- #
#  Runtime helpers (delegated to the real torch)                              #
# --------------------------------------------------------------------------- #
def use_deterministic_algorithms(
    mode: bool,
    *,
    warn_only: bool | None = ...,
) -> None: ...
def equal(a: Tensor, b: Tensor, /) -> bool: ...
