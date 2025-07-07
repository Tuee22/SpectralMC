# src/spectralmc/models/torch.py
"""
spectralmc.models.torch
=======================

A *minimal‑surface* façade exposing only the PyTorch functionality required by
SpectralMC while enforcing **full determinism**.

---------------------------------------------------------------------------
Why a façade?
---------------------------------------------------------------------------
* Central place to pin all reproducibility flags (cuBLAS, cuDNN, TF32, etc.).
* Provide strongly‑typed helpers (`DType`, `Device`) that round‑trip cleanly
  with the real PyTorch singletons **without** leaking dynamic types into the
  public API.
* Supply thread‑safe context‑managers for temporarily overriding the global
  default *dtype* / *device*.  The implementation serialises callers with
  two distinct locks, so a single thread restoring one setting never tramples
  the other.
* Offer a *portable* yet *loss‑less* representation of tensors and Adam
  optimiser state that can be shipped over the wire as plain JSON / msg‑pack.

This module is meant to be imported **before any third‑party code touches
`torch`**, guaranteeing that every subsequent tensor operation in the
interpreter inherits the deterministic configuration.
"""
from __future__ import annotations

###############################################################################
# Reproducibility toggles (MUST precede the first ``import torch``) -----------
###############################################################################
import os
import platform
import threading
from contextlib import contextmanager
from enum import Enum
from io import BytesIO
from typing import Dict, Iterable, Iterator, List, Mapping, Protocol, Tuple

# CuBLAS workspace size must be fixed *before* CUDA libraries are loaded.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")

# Import torch **after** the env‑var is set
import torch  # noqa: E402

###############################################################################
# Sanity checks – abort early if the runtime is not capable of deterministic
# CUDA execution (SpectralMC is GPU‑only by definition).
###############################################################################
if not torch.cuda.is_available():
    raise RuntimeError(
        "SpectralMC requires a CUDA‑enabled build of PyTorch, "
        "but torch.cuda.is_available() returned False.  "
        "Install the correct wheel or set CUDA_VISIBLE_DEVICES."
    )

if not hasattr(torch.backends, "cudnn") or torch.backends.cudnn.version() is None:
    raise RuntimeError(
        "SpectralMC needs cuDNN for deterministic convolutions, "
        "but it is not present in this runtime."
    )

###############################################################################
# Determinism knobs – no compromises (speed vs. reproducibility). -------------
###############################################################################
torch.use_deterministic_algorithms(True, warn_only=False)

# cuDNN flags
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# Disable all TF32 execution paths (they break bit‑wise reproducibility).
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

###############################################################################
# Strongly‑typed helpers ------------------------------------------------------
###############################################################################
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


class DType(str, Enum):
    """Floating‑point and complex formats accepted by SpectralMC."""

    float16 = "float16"
    float32 = "float32"
    float64 = "float64"
    bfloat16 = "bfloat16"
    complex64 = "complex64"
    complex128 = "complex128"

    # ------------------------------------------------------------------ #
    def to_torch(self) -> torch.dtype:
        """Return the matching ``torch.dtype`` singleton."""
        return _DTYPE_STR_TO_TORCH[self.value]

    @classmethod
    def from_torch(cls, dt: torch.dtype) -> "DType":
        """Convert a ``torch.dtype`` back to :class:`DType`."""
        try:
            return cls(_TORCH_DTYPE_TO_STR[dt])
        except KeyError as exc:  # pragma: no cover
            raise ValueError(f"Unsupported torch.dtype {dt!r}") from exc


class Device(str, Enum):
    """CPU or the *first* CUDA card (SpectralMC is single‑GPU)."""

    cpu = "cpu"
    cuda = "cuda:0"

    # ------------------------------------------------------------------ #
    def to_torch(self) -> torch.device:
        """Return the equivalent ``torch.device``."""
        return torch.device(self.value)

    @classmethod
    def from_torch(cls, dev: torch.device) -> "Device":
        """
        Convert a ``torch.device`` back to :class:`Device`.

        Only ``cpu`` and ``cuda:0`` are allowed.  Using any other CUDA index
        would void reproducibility guarantees because memory allocations are no
        longer deterministic across topology / peer‑to‑peer paths.
        """
        if dev.type == "cpu":
            return cls.cpu
        if dev.type == "cuda" and dev.index in {None, 0}:
            return cls.cuda
        raise ValueError(f"SpectralMC runs *only* on the first GPU, received {dev!r}.")


###############################################################################
# Thread‑safe context‑managers ------------------------------------------------
###############################################################################
_dtype_lock = threading.RLock()
_device_lock = threading.RLock()


@contextmanager
def default_dtype(dt: torch.dtype) -> Iterator[None]:
    """
    Temporarily override PyTorch's **process‑global** default dtype.

    The implementation serialises callers with a re‑entrant lock to avoid the
    classic race condition:

    * Thread A sets dtype→float32
    * Thread B sets dtype→float16
    * Thread A restores dtype to *its* previous value, clobbering B

    Using an RLock guarantees *LIFO* semantics and correctness even when the
    same thread nests the context‑manager recursively.

    Args:
        dt: Target `torch.dtype`.
    """
    with _dtype_lock:
        prev = torch.get_default_dtype()
        torch.set_default_dtype(dt)
        try:
            yield
        finally:
            torch.set_default_dtype(prev)


@contextmanager
def default_device(dev: torch.device) -> Iterator[None]:
    """
    Temporarily override PyTorch's **process‑global** default device.

    A dedicated lock is used rather than sharing `_dtype_lock` so that
    `with default_dtype(...), default_device(...):` does not dead‑lock due to
    lock re‑acquisition ordering.

    Args:
        dev: Target `torch.device`.
    """
    with _device_lock:
        prev = torch.tensor([]).device
        torch.set_default_device(dev)
        try:
            yield
        finally:
            torch.set_default_device(prev)


###############################################################################
# SafeTensor serialisation ----------------------------------------------------
###############################################################################
from pydantic import BaseModel, ConfigDict  # noqa: E402
from safetensors import safe_open          # noqa: E402
from safetensors.torch import save as _sf_save  # noqa: E402


class TensorState(BaseModel):
    """
    A CPU‑only SafeTensor snapshot that travels as raw bytes yet remembers
    shape and dtype for integrity checks on deserialisation.
    """

    data: bytes
    shape: Tuple[int, ...]
    dtype: DType

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=False)

    # ------------------------------------------------------------------ #
    @staticmethod
    def from_torch(t: torch.Tensor) -> "TensorState":
        """Create a snapshot from a **CPU** tensor."""
        if t.device != Device.cpu.to_torch():
            raise RuntimeError("TensorState expects a CPU tensor.")
        blob = _sf_save({"tensor": t})
        return TensorState(
            data=blob, shape=tuple(t.shape), dtype=DType.from_torch(t.dtype)
        )

    def to_torch(self) -> torch.Tensor:
        """Round‑trip back to a CPU tensor and verify integrity."""
        with safe_open(
            BytesIO(self.data), framework="pt", device=Device.cpu.value
        ) as reader:
            tensor = reader.get_tensor("tensor")

        if (
            tuple(tensor.shape) != self.shape
            or DType.from_torch(tensor.dtype) != self.dtype
        ):
            raise RuntimeError("Tensor metadata mismatch after deserialisation.")
        return tensor

    # Convenience constructor --------------------------------------------------
    @staticmethod
    def from_bytes(raw: bytes) -> "TensorState":
        """Validate arbitrary bytes and return a :class:`TensorState`."""
        with safe_open(BytesIO(raw), framework="pt", device=Device.cpu.value) as reader:
            if reader.keys() != ["tensor"]:
                raise ValueError("SafeTensor must contain exactly one entry 'tensor'.")
            t = reader.get_tensor("tensor")
        return TensorState(
            data=raw, shape=tuple(t.shape), dtype=DType.from_torch(t.dtype)
        )


###############################################################################
# Runtime fingerprint ---------------------------------------------------------
###############################################################################
class TorchEnv(BaseModel):
    """
    Structured snapshot of the active PyTorch / CUDA environment.

    The capture **fails** whenever CUDA or cuDNN information is missing,
    aligning with SpectralMC's reproducibility policy that mandates GPU
    execution.
    """

    torch_version: str
    cuda_version: str
    cudnn_version: int
    gpu_name: str
    python_version: str

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=False)

    # ------------------------------------------------------------------ #
    @classmethod
    def snapshot(cls) -> "TorchEnv":
        """Capture the current process' CUDA & PyTorch state."""
        return cls(
            torch_version=torch.__version__,
            cuda_version=torch.version.cuda or "<unknown>",
            cudnn_version=torch.backends.cudnn.version() or -1,
            gpu_name=torch.cuda.get_device_name(0),
            python_version=platform.python_version(),
        )


###############################################################################
# Adam optimiser helpers ------------------------------------------------------
###############################################################################
class _HasStateDict(Protocol):
    """Any optimiser exposing exactly ``state_dict()``."""

    def state_dict(self) -> Mapping[str, object]: ...


class AdamParamState(BaseModel):
    """Per‑parameter state required by Adam (and AdamW)."""

    step: int
    exp_avg: TensorState
    exp_avg_sq: TensorState
    max_exp_avg_sq: TensorState | None

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=False)

    # ------------------------------------------------------------------ #
    @classmethod
    def from_torch(cls, s: Mapping[str, object]) -> "AdamParamState":
        """Validate a raw PyTorch param‑state mapping and convert it."""
        allowed = {"step", "exp_avg", "exp_avg_sq", "max_exp_avg_sq"}
        unexpected = set(s) - allowed
        if unexpected:  # pragma: no cover
            raise RuntimeError(f"Unexpected key(s) in Adam state: {unexpected}")

        def _req_tensor(obj: object, name: str) -> torch.Tensor:
            if not isinstance(obj, torch.Tensor):
                raise TypeError(f"Adam '{name}' must be torch.Tensor.")
            if obj.device != Device.cpu.to_torch():
                raise RuntimeError("All Adam state tensors must reside on CPU.")
            return obj

        # ----- strict type‑check for 'step' (fixes mypy overload error) -----
        step_raw = s["step"]
        if not isinstance(step_raw, int):
            raise TypeError("Adam 'step' must be int.")

        return cls(
            step=step_raw,
            exp_avg=TensorState.from_torch(_req_tensor(s["exp_avg"], "exp_avg")),
            exp_avg_sq=TensorState.from_torch(
                _req_tensor(s["exp_avg_sq"], "exp_avg_sq")
            ),
            max_exp_avg_sq=(
                TensorState.from_torch(
                    _req_tensor(s["max_exp_avg_sq"], "max_exp_avg_sq")
                )
                if s.get("max_exp_avg_sq") is not None
                else None
            ),
        )

    # ------------------------------------------------------------------ #
    def to_torch(self) -> Dict[str, object]:
        """Return PyTorch's plain, CPU‑tensor mapping."""
        base: Dict[str, object] = {
            "step": self.step,
            "exp_avg": self.exp_avg.to_torch(),
            "exp_avg_sq": self.exp_avg_sq.to_torch(),
        }
        if self.max_exp_avg_sq is not None:
            base["max_exp_avg_sq"] = self.max_exp_avg_sq.to_torch()
        return base


class AdamParamGroup(BaseModel):
    """One parameter group inside an Adam(W) optimiser."""

    params: List[int]
    lr: float
    betas: Tuple[float, float]
    eps: float
    weight_decay: float
    amsgrad: bool = False
    maximize: bool = False

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=False)

    @classmethod
    def from_torch(cls, g: Mapping[str, object]) -> "AdamParamGroup":
        """Validate and convert a raw PyTorch param‑group."""
        return cls.model_validate(g)

    def to_torch(self) -> Dict[str, object]:
        """Return a plain mapping suitable for PyTorch."""
        return self.model_dump(mode="python")


class AdamOptimizerState(BaseModel):
    """
    Fully‑typed, serialisable capture of an Adam / AdamW optimiser.

    Shape matches ``torch.optim.Adam.state_dict()`` exactly, but every nested
    node is type‑checked and safe to (de)serialise with JSON / msg‑pack.
    """

    param_states: Dict[int, AdamParamState]
    param_groups: List[AdamParamGroup]

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=False)

    # ------------------------------------------------------------------ #
    @classmethod
    def from_torch(cls, optim: _HasStateDict) -> "AdamOptimizerState":
        """Validate *optim* and convert its state‑dict."""
        if optim.__class__.__name__ not in {"Adam", "AdamW"}:
            raise TypeError("Only torch.optim.Adam or AdamW are supported.")

        sd = optim.state_dict()
        if set(sd) != {"state", "param_groups"}:  # pragma: no cover
            raise RuntimeError(
                "state_dict() must contain exactly {'state', 'param_groups'}."
            )

        state_raw, groups_raw = sd["state"], sd["param_groups"]

        if not isinstance(state_raw, Mapping):
            raise TypeError("'state' must be a mapping.")
        if not isinstance(groups_raw, Iterable):
            raise TypeError("'param_groups' must be a sequence.")

        return cls(
            param_states={
                pid: AdamParamState.from_torch(st) for pid, st in state_raw.items()
            },
            param_groups=[AdamParamGroup.from_torch(pg) for pg in groups_raw],
        )

    # ------------------------------------------------------------------ #
    def to_torch(self) -> Dict[str, object]:
        """Reverse the transformation back to PyTorch's consumable dict."""
        return {
            "state": {pid: st.to_torch() for pid, st in self.param_states.items()},
            "param_groups": [g.to_torch() for g in self.param_groups],
        }


###############################################################################
# Public re‑exports – the only names that should leak out of this module ------
###############################################################################
__all__: Tuple[str, ...] = (
    "DType",
    "Device",
    "TensorState",
    "TorchEnv",
    "AdamParamState",
    "AdamParamGroup",
    "AdamOptimizerState",
    "default_dtype",
    "default_device",
)
