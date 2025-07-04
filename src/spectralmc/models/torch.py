# src/spectralmc/models/torch.py
"""
spectralmc.models.torch
=======================

Typed‑pure façade around the tiny PyTorch surface that SpectralMC needs.
The file *passes* ``mypy --strict`` with **zero** suppressions.

Key features
------------
* **Determinism** – cuBLAS / cuDNN are pinned to reproducible behaviour.
* :class:`dtype` and :class:`device` enums that convert loss‑lessly to and
  from the real ``torch`` singletons.
* Two context‑managers (:func:`default_dtype`, :func:`default_device`)
  mirroring PyTorch helpers while keeping the type‑checker happy.
* :class:`TensorState` – a CPU‑only SafeTensors snapshot that travels as
  raw bytes yet retains shape / dtype metadata.
* :class:`TorchEnv` – a structured CUDA / PyTorch fingerprint.
* A fully‑typed capture of Adam(W) optimiser state that is round‑trippable
  via JSON / msg‑pack.

Everything here rejects tensors on non‑CPU devices when serialising (the
SafeTensors archive itself is host‑only).
"""
from __future__ import annotations

import os
import platform
from contextlib import contextmanager
from enum import Enum
from io import BytesIO
from typing import Dict, Iterable, Iterator, List, Mapping, Protocol, Tuple

import torch
from pydantic import BaseModel, ConfigDict
from safetensors import safe_open
from safetensors.torch import save as _sf_save

__all__: Tuple[str, ...] = (
    "dtype",
    "device",
    "TensorState",
    "TorchEnv",
    "AdamParamState",
    "AdamParamGroup",
    "AdamOptimizerState",
    "default_dtype",
    "default_device",
)

# ───────────────────── reproducibility toggles ──────────────────────────────
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")
torch.use_deterministic_algorithms(True, warn_only=False)
if hasattr(torch.backends, "cudnn"):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ───────────────────────────── dtype helpers ────────────────────────────────
_DTYPE_STR_TO_TORCH: Dict[str, torch.dtype] = {
    "float16": torch.float16,
    "float32": torch.float32,
    "float64": torch.float64,
    "bfloat16": torch.bfloat16,
    "complex64": torch.complex64,
    "complex128": torch.complex128,
}
_TORCH_DTYPE_TO_STR: Dict[torch.dtype, str] = {
    v: k for k, v in _DTYPE_STR_TO_TORCH.items()
}


class dtype(str, Enum):
    """Floating‑point and complex formats SpectralMC accepts."""

    float16 = "float16"
    float32 = "float32"
    float64 = "float64"
    bfloat16 = "bfloat16"
    complex64 = "complex64"
    complex128 = "complex128"

    # --------------------------------------------------------------------- #
    # Converters
    # --------------------------------------------------------------------- #
    def to_torch(self) -> torch.dtype:
        """Return the matching ``torch.dtype`` singleton."""
        return _DTYPE_STR_TO_TORCH[self.value]

    @classmethod
    def from_torch(cls, dt: torch.dtype) -> "dtype":
        """Convert a ``torch.dtype`` back to :class:`dtype`."""
        try:
            return cls(_TORCH_DTYPE_TO_STR[dt])
        except KeyError as exc:  # pragma: no cover
            raise ValueError(f"Unsupported torch.dtype {dt!r}") from exc


# ───────────────────────────── device helpers ───────────────────────────────
class device(str, Enum):
    """Either host CPU or the first CUDA card."""

    cpu = "cpu"
    cuda = "cuda:0"

    def to_torch(self) -> torch.device:
        """Return the equivalent ``torch.device``."""
        return torch.device(self.value)

    @classmethod
    def from_torch(cls, dev: torch.device) -> "device":
        """Convert a ``torch.device`` back to :class:`device`."""
        return cls(dev.type)


# ───────────────────────────── context‑managers ─────────────────────────────
@contextmanager
def default_dtype(dt: torch.dtype) -> Iterator[None]:
    """Temporarily override PyTorch’s global default dtype."""
    prev = torch.get_default_dtype()
    torch.set_default_dtype(dt)
    try:
        yield
    finally:
        torch.set_default_dtype(prev)


@contextmanager
def default_device(dev: torch.device) -> Iterator[None]:
    """Temporarily override PyTorch’s global default device."""
    prev = torch.tensor([]).device
    torch.set_default_device(dev)
    try:
        yield
    finally:
        torch.set_default_device(prev)


# ───────────────────────── Tensor serialisation ─────────────────────────────
class TensorState(BaseModel):
    """CPU‑only SafeTensors payload plus shape / dtype metadata."""

    data: bytes
    shape: Tuple[int, ...]
    dtype: dtype

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=False)

    # ------------------------------------------------------------------ #
    @staticmethod
    def from_torch(t: torch.Tensor) -> "TensorState":
        """Snapshot *t* (must live on CPU) into an in‑memory SafeTensor."""
        if t.device != device.cpu.to_torch():
            raise RuntimeError("TensorState expects a **CPU** tensor.")
        blob = _sf_save({"tensor": t})
        return TensorState(
            data=blob, shape=tuple(t.shape), dtype=dtype.from_torch(t.dtype)
        )

    def to_torch(self) -> torch.Tensor:
        """Materialise the tensor back on CPU and verify integrity."""
        with safe_open(
            BytesIO(self.data), framework="pt", device=device.cpu.value
        ) as reader:
            tensor = reader.get_tensor("tensor")
        if (
            tuple(tensor.shape) != self.shape
            or dtype.from_torch(tensor.dtype) != self.dtype
        ):
            raise RuntimeError("Stored metadata does not match tensor contents.")
        return tensor

    @staticmethod
    def from_bytes(raw: bytes) -> "TensorState":
        """Validate *raw* bytes and build a :class:`TensorState`."""
        with safe_open(BytesIO(raw), framework="pt", device=device.cpu.value) as reader:
            if reader.keys() != ["tensor"]:
                raise ValueError(
                    "SafeTensor must contain exactly one entry named 'tensor'."
                )
            t = reader.get_tensor("tensor")
        return TensorState(
            data=raw, shape=tuple(t.shape), dtype=dtype.from_torch(t.dtype)
        )


# ───────────────────────────── runtime fingerprint ──────────────────────────
class TorchEnv(BaseModel):
    """Concise snapshot of the active PyTorch / CUDA environment."""

    torch_version: str
    cuda_version: str | None
    cudnn_version: int | None
    gpu_name: str
    python_version: str

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=False)

    # ------------------------------------------------------------------ #
    @classmethod
    def snapshot(cls) -> "TorchEnv":
        """Capture the current process’ CUDA & PyTorch state."""
        return cls(
            torch_version=torch.__version__,
            cuda_version=torch.version.cuda,
            cudnn_version=torch.backends.cudnn.version(),
            gpu_name=torch.cuda.get_device_name(0),
            python_version=platform.python_version(),
        )


# ───────────────────────────── Adam helpers ────────────────────────────────
class _HasStateDict(Protocol):
    """Optimisers exposing *exactly* ``state_dict()``."""

    def state_dict(self) -> Mapping[str, object]: ...


class AdamParamState(BaseModel):
    """Per‑parameter state required by Adam(W)."""

    step: int
    exp_avg: TensorState
    exp_avg_sq: TensorState
    max_exp_avg_sq: TensorState | None

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=False)

    # ------------------------------------------------------------------ #
    @classmethod
    def from_torch(cls, s: Mapping[str, object]) -> "AdamParamState":
        """Convert PyTorch’s raw state‑mapping to a typed snapshot."""
        allowed = {"step", "exp_avg", "exp_avg_sq", "max_exp_avg_sq"}
        unexpected = set(s) - allowed
        if unexpected:  # pragma: no cover
            raise RuntimeError(f"Unexpected key(s) in Adam state: {unexpected}")

        step_obj = s["step"]
        if not isinstance(step_obj, int):
            raise TypeError("Adam 'step' must be int.")

        def _req_tensor(obj: object, name: str) -> torch.Tensor:
            if not isinstance(obj, torch.Tensor):
                raise TypeError(f"Adam '{name}' must be torch.Tensor.")
            if obj.device != device.cpu.to_torch():
                raise RuntimeError("All Adam state tensors must reside on CPU.")
            return obj

        exp_avg_t = _req_tensor(s["exp_avg"], "exp_avg")
        exp_avg_sq_t = _req_tensor(s["exp_avg_sq"], "exp_avg_sq")

        max_raw = s.get("max_exp_avg_sq")
        max_t = _req_tensor(max_raw, "max_exp_avg_sq") if max_raw is not None else None

        return cls(
            step=step_obj,
            exp_avg=TensorState.from_torch(exp_avg_t),
            exp_avg_sq=TensorState.from_torch(exp_avg_sq_t),
            max_exp_avg_sq=(
                TensorState.from_torch(max_t) if max_t is not None else None
            ),
        )

    # ------------------------------------------------------------------ #
    def to_torch(self) -> Dict[str, object]:
        """Return PyTorch’s plain, CPU‑tensor mapping."""
        base: Dict[str, object] = {
            "step": self.step,
            "exp_avg": self.exp_avg.to_torch(),
            "exp_avg_sq": self.exp_avg_sq.to_torch(),
        }
        if self.max_exp_avg_sq is not None:
            base["max_exp_avg_sq"] = self.max_exp_avg_sq.to_torch()
        return base


class AdamParamGroup(BaseModel):
    """A single parameter‑group within an Adam(W) optimiser."""

    params: List[int]
    lr: float
    betas: Tuple[float, float]
    eps: float
    weight_decay: float
    amsgrad: bool = False
    maximize: bool = False

    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=False)

    @classmethod
    def from_torch(cls, g: Mapping[str, object]) -> "AdamParamGroup":
        """Type‑check *g* via Pydantic validation."""
        return cls.model_validate(g)

    def to_torch(self) -> Dict[str, object]:
        """Return a plain mapping suitable for PyTorch."""
        return self.model_dump(mode="python")


class AdamOptimizerState(BaseModel):
    """
    Fully‑typed, serialisable capture of an Adam / AdamW optimiser.

    Shape matches ``torch.optim.Adam.state_dict()`` exactly, but every
    nested piece is validated and safe to serialise.
    """

    param_states: Dict[int, AdamParamState]
    param_groups: List[AdamParamGroup]

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=False)

    # ------------------------------------------------------------------ #
    @classmethod
    def from_torch(cls, optim: _HasStateDict) -> "AdamOptimizerState":
        """Validate *optim* and convert its state‑dict."""
        if optim.__class__.__name__ not in {"Adam", "AdamW"}:
            raise TypeError("Only Adam or AdamW are supported.")

        sd = optim.state_dict()
        if set(sd) != {"state", "param_groups"}:  # pragma: no cover
            raise RuntimeError(
                "state_dict() must contain exactly {'state', 'param_groups'}."
            )

        state_raw = sd["state"]
        if not isinstance(state_raw, Mapping):
            raise TypeError("'state' must be a mapping.")

        groups_raw = sd["param_groups"]
        if not isinstance(groups_raw, Iterable):
            raise TypeError("'param_groups' must be a sequence.")

        param_states = {
            pid: AdamParamState.from_torch(st) for pid, st in state_raw.items()
        }
        param_groups = [AdamParamGroup.from_torch(pg) for pg in groups_raw]
        return cls(param_states=param_states, param_groups=param_groups)

    # ------------------------------------------------------------------ #
    def to_torch(self) -> Dict[str, object]:
        """Reverse the transformation back to PyTorch’s consumable dict."""
        return {
            "state": {pid: st.to_torch() for pid, st in self.param_states.items()},
            "param_groups": [g.to_torch() for g in self.param_groups],
        }
