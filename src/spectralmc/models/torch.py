# src/spectralmc/models/torch.py
"""
spectralmc.models.torch
=======================

A minimal, fully-typed façade exposing just enough of PyTorch for SpectralMC
while enforcing reproducible execution.

Thread-safety contract
----------------------
* This module **must be imported from the main thread _before_ any other
  threads are spawned**; otherwise it raises ``ImportError``.
* The context managers ``default_dtype`` and ``default_device`` **may only be
  entered from the main thread**; attempting to use them from a worker thread
  raises ``RuntimeError``.
* To override the import-time thread guard (at your own risk) set the
  environment variable::

      SPECTRALMC_ALLOW_THREADS=1
"""

from __future__ import annotations

# --------------------------------------------------------------------------- #
#  Early import-time guards                                                   #
# --------------------------------------------------------------------------- #
import os
import platform
import sys
import threading
from contextlib import contextmanager
from enum import Enum
from typing import Iterable, Iterator, Mapping


# --- prevent pre-emptive torch import -------------------------------------- #
if "torch" in sys.modules:
    raise ImportError(
        "PyTorch was imported before the SpectralMC torch façade. "
        "Import 'spectralmc.models.torch' **first** so it can set "
        "deterministic flags."
    )

# --- thread-safety: import must occur before worker threads ---------------- #
if threading.active_count() > 1:
    raise ImportError(
        "SpectralMC façade imported after additional threads were created. "
        "Its global dtype/device helpers are not thread-safe. "
        "Import this module in the *main thread* before spawning workers, "
        "or set SPECTRALMC_ALLOW_THREADS=1 to bypass this check."
    )

_MAIN_THREAD_ID: int = threading.get_ident()

# --------------------------------------------------------------------------- #
#  Early environment fixes                                                    #
# --------------------------------------------------------------------------- #
# Ensure deterministic cuBLAS kernels (ignored if CUDA is absent)
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")

import torch  # noqa: E402

from spectralmc.errors.torch_facade import (  # noqa: E402
    InvalidAdamState,
    TensorStateConversionFailed,
    TorchFacadeResult,
    UnsupportedTorchDevice,
    UnsupportedTorchDType,
)
from spectralmc.result import Failure, Success  # noqa: E402
from spectralmc.models.numerical import Precision  # noqa: E402


# --------------------------------------------------------------------------- #
#  Determinism knobs                                                          #
# --------------------------------------------------------------------------- #
torch.use_deterministic_algorithms(True, warn_only=False)

# NOTE: _HAS_CUDA is infrastructure for module-level cuDNN configuration.
# This is acceptable per CPU/GPU policy: facade code must check CUDA availability
# at import time to configure determinism flags. It does NOT create silent
# fallbacks - all GPU operations in SpectralMC fail-fast via explicit device
# assertions in the compute layer (see cpu_gpu_compute_policy.md).
_HAS_CUDA: bool = torch.cuda.is_available()

if _HAS_CUDA:
    if torch.backends.cudnn.version() is None:
        raise RuntimeError(
            "SpectralMC requires cuDNN for deterministic GPU execution, "
            "but it is missing from the current runtime.",
        )
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
else:
    # Even on CPU-only builds make sure these flags exist and are disabled
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False


# --------------------------------------------------------------------------- #
#  Helper: assert main thread                                                 #
# --------------------------------------------------------------------------- #
def _assert_main_thread(ctx_name: str) -> None:
    """Raise if called from any thread other than the importing main thread."""
    if threading.get_ident() != _MAIN_THREAD_ID:
        raise RuntimeError(f"{ctx_name} is not thread-safe; call it only from the main thread.")


# --------------------------------------------------------------------------- #
#  Strongly-typed helpers                                                     #
# --------------------------------------------------------------------------- #
_DTYPE_STR_TO_TORCH: dict[str, torch.dtype] = {
    "float16": torch.float16,
    "float32": torch.float32,
    "float64": torch.float64,
    "bfloat16": torch.bfloat16,
    "complex64": torch.complex64,
    "complex128": torch.complex128,
}
_TORCH_DTYPE_TO_STR: dict[torch.dtype, str] = {v: k for k, v in _DTYPE_STR_TO_TORCH.items()}
_DTYPE_STR_TO_PRECISION: dict[str, Precision] = {
    "float32": Precision.float32,
    "float64": Precision.float64,
    "complex64": Precision.complex64,
    "complex128": Precision.complex128,
}
_PRECISION_DTYPE_TO_STR: dict[Precision, str] = {v: k for k, v in _DTYPE_STR_TO_PRECISION.items()}


class FullPrecisionDType(str, Enum):
    """DTypes with corresponding Precision representation (for MC simulation)."""

    float32 = "float32"
    float64 = "float64"
    complex64 = "complex64"
    complex128 = "complex128"

    # --- conversion helpers -------------------------------------------------
    def to_torch(self) -> torch.dtype:
        """Return the corresponding ``torch.dtype``."""
        return _DTYPE_STR_TO_TORCH[self.value]

    @classmethod
    def from_torch(cls, dt: torch.dtype) -> TorchFacadeResult["FullPrecisionDType"]:
        dtype_str = _TORCH_DTYPE_TO_STR.get(dt)
        if dtype_str not in ("float32", "float64", "complex64", "complex128"):
            return Failure(UnsupportedTorchDType(dtype=str(dt)))
        return Success(cls(dtype_str))

    def to_precision(self) -> Precision:
        """Return the numeric ``Precision`` representation of this dtype."""
        return _DTYPE_STR_TO_PRECISION[self.value]

    @classmethod
    def from_precision(cls, p: Precision) -> FullPrecisionDType:
        return cls(_PRECISION_DTYPE_TO_STR[p])


class ReducedPrecisionDType(str, Enum):
    """DTypes without Precision representation (for storage/mixed-precision training)."""

    float16 = "float16"
    bfloat16 = "bfloat16"

    # --- conversion helpers -------------------------------------------------
    def to_torch(self) -> torch.dtype:
        """Return the corresponding ``torch.dtype``."""
        return _DTYPE_STR_TO_TORCH[self.value]

    @classmethod
    def from_torch(cls, dt: torch.dtype) -> TorchFacadeResult["ReducedPrecisionDType"]:
        dtype_str = _TORCH_DTYPE_TO_STR.get(dt)
        if dtype_str not in ("float16", "bfloat16"):
            return Failure(UnsupportedTorchDType(dtype=str(dt)))
        return Success(cls(dtype_str))


# Union type for contexts accepting any dtype
AnyDType = FullPrecisionDType | ReducedPrecisionDType

# Backward compatibility alias (deprecated, prefer explicit types)
DType = FullPrecisionDType


class Device(str, Enum):
    cpu = "cpu"
    cuda = "cuda:0"  # single-GPU policy

    # --- conversion helpers -------------------------------------------------
    def to_torch(self) -> torch.device:
        """Return the corresponding ``torch.device``."""
        return torch.device(self.value)

    @classmethod
    def from_torch(cls, dev: torch.device) -> TorchFacadeResult["Device"]:
        if dev.type == "cpu":
            return Success(cls.cpu)
        if dev.type == "cuda" and dev.index in {None, 0}:
            return Success(cls.cuda)
        return Failure(UnsupportedTorchDevice(device=str(dev)))


# --------------------------------------------------------------------------- #
#  Lightweight global context managers (main-thread only)                     #
# --------------------------------------------------------------------------- #
@contextmanager
def default_dtype(dt: torch.dtype) -> Iterator[None]:
    """Temporarily sets the global default dtype (main thread only)."""
    _assert_main_thread("default_dtype")
    _tid = threading.get_ident()
    previous = torch.get_default_dtype()
    torch.set_default_dtype(dt)
    try:
        yield
    finally:
        if threading.get_ident() != _tid:  # sanity: same thread exits
            raise RuntimeError(
                "default_dtype context exited in a different thread than it entered."
            )
        torch.set_default_dtype(previous)


@contextmanager
def default_device(dev: torch.device) -> Iterator[None]:
    """Temporarily sets the global default device (main thread only)."""
    _assert_main_thread("default_device")
    _tid = threading.get_ident()
    previous = torch.tensor([]).device
    torch.set_default_device(dev)
    try:
        yield
    finally:
        if threading.get_ident() != _tid:
            raise RuntimeError(
                "default_device context exited in a different thread than it entered."
            )
        torch.set_default_device(previous)


# --------------------------------------------------------------------------- #
#  SafeTensor serialization helpers                                           #
# --------------------------------------------------------------------------- #
from pydantic import BaseModel, ConfigDict  # noqa: E402
from safetensors.torch import load as _sf_load  # noqa: E402
from safetensors.torch import save as _sf_save  # noqa: E402


class TensorState(BaseModel):
    """A CPU-only SafeTensor snapshot with integrity metadata."""

    data: bytes
    shape: tuple[int, ...]
    dtype: AnyDType

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=False)

    # ------------------------------------------------------------------ #
    @staticmethod
    def from_torch(t: torch.Tensor) -> TorchFacadeResult["TensorState"]:
        if t.device != Device.cpu.to_torch():
            return Failure(
                TensorStateConversionFailed(
                    message="Tensor must reside on CPU to enter TensorState."
                )
            )
        blob: bytes = _sf_save({"tensor": t})

        dt_torch = t.dtype
        dtype_str = _TORCH_DTYPE_TO_STR.get(dt_torch)
        if dtype_str is None:
            return Failure(
                TensorStateConversionFailed(message=f"Unsupported torch.dtype {dt_torch!r}")
            )

        dtype: AnyDType
        if dtype_str in ("float32", "float64", "complex64", "complex128"):
            dtype = FullPrecisionDType(dtype_str)
        else:
            dtype = ReducedPrecisionDType(dtype_str)

        return Success(TensorState(data=blob, shape=tuple(t.shape), dtype=dtype))

    def to_torch(self) -> TorchFacadeResult[torch.Tensor]:
        tensor_dict: dict[str, torch.Tensor] = _sf_load(self.data)
        tensor = tensor_dict.get("tensor")
        if tensor is None:
            return Failure(
                TensorStateConversionFailed(
                    message="SafeTensor payload missing required 'tensor' entry."
                )
            )
        if tensor.device != Device.cpu.to_torch():
            return Failure(
                TensorStateConversionFailed(message="Tensor deserialized onto a non-CPU device.")
            )

        expected_dtype = self.dtype.to_torch()
        if tuple(tensor.shape) != self.shape or tensor.dtype != expected_dtype:
            return Failure(
                TensorStateConversionFailed(message="Tensor metadata mismatch on deserialization.")
            )
        return Success(tensor)

    @staticmethod
    def from_bytes(raw: bytes) -> TorchFacadeResult["TensorState"]:
        tensor_dict: dict[str, torch.Tensor] = _sf_load(raw)
        if set(tensor_dict.keys()) != {"tensor"}:
            return Failure(
                TensorStateConversionFailed(
                    message="SafeTensor must contain exactly one entry named 'tensor'."
                )
            )
        t = tensor_dict["tensor"]
        if t.device != Device.cpu.to_torch():
            return Failure(
                TensorStateConversionFailed(message="Tensor deserialized onto a non-CPU device.")
            )

        dt_torch = t.dtype
        dtype_str = _TORCH_DTYPE_TO_STR.get(dt_torch)
        if dtype_str is None:
            return Failure(
                TensorStateConversionFailed(message=f"Unsupported torch.dtype {dt_torch!r}")
            )

        dtype: AnyDType
        if dtype_str in ("float32", "float64", "complex64", "complex128"):
            dtype = FullPrecisionDType(dtype_str)
        else:
            dtype = ReducedPrecisionDType(dtype_str)

        return Success(TensorState(data=raw, shape=tuple(t.shape), dtype=dtype))


# --------------------------------------------------------------------------- #
#  Environment fingerprint                                                    #
# --------------------------------------------------------------------------- #
class TorchEnv(BaseModel):
    torch_version: str
    cuda_version: str
    cudnn_version: int
    gpu_name: str
    python_version: str

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=False)

    @classmethod
    def snapshot(cls) -> TorchEnv:
        if _HAS_CUDA:
            cuda_ver = torch.version.cuda or "<unknown>"
            cudnn_ver = torch.backends.cudnn.version() or -1
            gpu = torch.cuda.get_device_name(0)
        else:
            cuda_ver = "<not available>"
            cudnn_ver = -1
            gpu = "<cpu>"
        return cls(
            torch_version=torch.__version__,
            cuda_version=cuda_ver,
            cudnn_version=cudnn_ver,
            gpu_name=gpu,
            python_version=platform.python_version(),
        )


# --------------------------------------------------------------------------- #
#  Adam optimizer helpers                                                     #
# --------------------------------------------------------------------------- #
class AdamParamState(BaseModel):
    step: int
    exp_avg: TensorState
    exp_avg_sq: TensorState
    max_exp_avg_sq: TensorState | None

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=False)

    # ------------------------------------------------------------------ #
    @classmethod
    def from_torch(cls, s: Mapping[str, object]) -> TorchFacadeResult["AdamParamState"]:
        allowed = {"step", "exp_avg", "exp_avg_sq", "max_exp_avg_sq"}
        extra = set(s) - allowed
        if extra:
            return Failure(InvalidAdamState(message=f"Unexpected Adam state keys: {extra}"))

        def _req_tensor(name: str) -> TorchFacadeResult[torch.Tensor]:
            obj = s[name]
            if not isinstance(obj, torch.Tensor):
                return Failure(InvalidAdamState(message=f"Adam '{name}' must be a torch.Tensor."))
            if obj.device != Device.cpu.to_torch():
                return Failure(InvalidAdamState(message="Adam state tensors must reside on CPU."))
            return Success(obj)

        step_raw = s["step"]
        if isinstance(step_raw, torch.Tensor):
            if step_raw.ndim != 0 or step_raw.device != Device.cpu.to_torch():
                return Failure(InvalidAdamState(message="Adam 'step' must be a CPU scalar tensor."))
            step_val = int(step_raw.item())
        elif isinstance(step_raw, int):
            step_val = step_raw
        else:
            return Failure(InvalidAdamState(message="Adam 'step' must be an int or scalar tensor."))

        match _req_tensor("exp_avg"):
            case Failure(error):
                return Failure(error)
            case Success(exp_avg_tensor):
                pass

        match _req_tensor("exp_avg_sq"):
            case Failure(error):
                return Failure(error)
            case Success(exp_avg_sq_tensor):
                pass

        max_tensor: torch.Tensor | None = None
        if s.get("max_exp_avg_sq") is not None:
            match _req_tensor("max_exp_avg_sq"):
                case Failure(error):
                    return Failure(error)
                case Success(max_value):
                    max_tensor = max_value

        match TensorState.from_torch(exp_avg_tensor):
            case Failure(error):
                return Failure(error)
            case Success(exp_avg_state):
                pass

        match TensorState.from_torch(exp_avg_sq_tensor):
            case Failure(error):
                return Failure(error)
            case Success(exp_avg_sq_state):
                pass

        max_state: TensorState | None = None
        if max_tensor is not None:
            match TensorState.from_torch(max_tensor):
                case Failure(error):
                    return Failure(error)
                case Success(state):
                    max_state = state

        return Success(
            cls(
                step=step_val,
                exp_avg=exp_avg_state,
                exp_avg_sq=exp_avg_sq_state,
                max_exp_avg_sq=max_state,
            )
        )

    def to_torch(self) -> TorchFacadeResult[dict[str, object]]:
        match self.exp_avg.to_torch():
            case Failure(error):
                return Failure(error)
            case Success(exp_avg_tensor):
                pass

        match self.exp_avg_sq.to_torch():
            case Failure(error):
                return Failure(error)
            case Success(exp_avg_sq_tensor):
                pass

        max_tensor: torch.Tensor | None = None
        if self.max_exp_avg_sq is not None:
            match self.max_exp_avg_sq.to_torch():
                case Failure(error):
                    return Failure(error)
                case Success(max_value):
                    max_tensor = max_value

        payload: dict[str, object] = {
            "step": self.step,
            "exp_avg": exp_avg_tensor,
            "exp_avg_sq": exp_avg_sq_tensor,
        }
        if max_tensor is not None:
            payload["max_exp_avg_sq"] = max_tensor
        return Success(payload)


class AdamParamGroup(BaseModel):
    params: list[int]
    lr: float
    betas: tuple[float, float]
    eps: float
    weight_decay: float
    amsgrad: bool = False
    maximize: bool = False
    foreach: bool | None = None
    capturable: bool = False
    differentiable: bool = False
    fused: bool | None = None
    decoupled_weight_decay: bool = False

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=False)

    @classmethod
    def from_torch(cls, g: Mapping[str, object]) -> AdamParamGroup:
        # Pydantic will both validate and populate all known fields
        return cls.model_validate(g)

    def to_torch(self) -> dict[str, object]:
        # Convert the Pydantic model to a Python dict for PyTorch consumption
        return self.model_dump(mode="python")


class AdamOptimizerState(BaseModel):
    """
    A structured, fully-typed representation of a PyTorch Adam/AdamW
    ``state_dict``.  The class no longer depends on an optimiser instance
    (and therefore no longer requires a ``_HasStateDict`` protocol); instead
    it directly consumes and produces plain ``Mapping[str, object]`` state
    dictionaries.
    """

    param_states: dict[int, AdamParamState]
    param_groups: list[AdamParamGroup]

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=False)

    # ------------------------------------------------------------------ #
    @classmethod
    def from_torch(cls, sd: Mapping[str, object]) -> TorchFacadeResult["AdamOptimizerState"]:
        """Construct from a PyTorch ``state_dict`` mapping."""
        if set(sd.keys()) != {"state", "param_groups"}:
            return Failure(
                InvalidAdamState(
                    message="state_dict must contain exactly the keys 'state' and 'param_groups'."
                )
            )

        state_raw = sd["state"]
        groups_raw = sd["param_groups"]

        if not isinstance(state_raw, Mapping):
            return Failure(InvalidAdamState(message="'state' entry must be a mapping."))
        if not isinstance(groups_raw, Iterable):
            return Failure(InvalidAdamState(message="'param_groups' entry must be a sequence."))

        param_states: dict[int, AdamParamState] = {}
        for pid, st in state_raw.items():
            state_result = AdamParamState.from_torch(st)
            match state_result:
                case Failure(error):
                    return Failure(error)
                case Success(param_state):
                    param_states[pid] = param_state

        param_groups: list[AdamParamGroup] = [AdamParamGroup.from_torch(pg) for pg in groups_raw]

        return Success(cls(param_states=param_states, param_groups=param_groups))

    # ------------------------------------------------------------------ #
    def to_torch(self) -> TorchFacadeResult[Mapping[str, object]]:
        serialized_states: dict[int, dict[str, object]] = {}
        for pid, param_state in self.param_states.items():
            match param_state.to_torch():
                case Failure(error):
                    return Failure(error)
                case Success(state_dict):
                    serialized_states[pid] = state_dict

        return Success(
            {
                "state": serialized_states,
                "param_groups": [g.to_torch() for g in self.param_groups],
            }
        )


# --------------------------------------------------------------------------- #
#  Public API                                                                 #
# --------------------------------------------------------------------------- #
__all__: tuple[str, ...] = (
    "FullPrecisionDType",
    "ReducedPrecisionDType",
    "AnyDType",
    "DType",  # Backward compatibility alias
    "Device",
    "TensorState",
    "TorchEnv",
    "AdamParamState",
    "AdamParamGroup",
    "AdamOptimizerState",
    "default_dtype",
    "default_device",
)
