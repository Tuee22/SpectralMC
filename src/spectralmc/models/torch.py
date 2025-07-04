# src/spectralmc/models/torch.py
from __future__ import annotations

import os
import platform
from contextlib import contextmanager
from enum import Enum
from io import BytesIO
from typing import Dict, Iterator, List, Mapping, Protocol, Tuple

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

# ─────────────────────── global reproducibility toggles ──────────────────────
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")
torch.use_deterministic_algorithms(True, warn_only=False)
if hasattr(torch.backends, "cudnn"):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ───────────────────────────── dtype infrastructure ──────────────────────────
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
    float16 = "float16"
    float32 = "float32"
    float64 = "float64"
    bfloat16 = "bfloat16"
    complex64 = "complex64"
    complex128 = "complex128"

    def to_torch(self) -> torch.dtype:
        return _DTYPE_STR_TO_TORCH[self.value]

    @classmethod
    def from_torch(cls, dt: torch.dtype) -> dtype:
        try:
            return cls(_TORCH_DTYPE_TO_STR[dt])
        except KeyError as exc:
            raise ValueError(f"Unsupported torch.dtype {dt!r}") from exc


# ─────────────────────────── device infrastructure ───────────────────────────
class device(str, Enum):
    cpu = "cpu"
    cuda = "cuda:0"  # generalize later if we need more than 1 cuda device

    def to_torch(self) -> torch.device:
        return torch.device(self.value)

    @classmethod
    def from_torch(cls, dev: torch.device) -> device:
        return cls(dev.type)


# ───────────────────────────── context managers ──────────────────────────────
@contextmanager
def default_dtype(dt: torch.dtype) -> Iterator[None]:
    prev = torch.get_default_dtype()
    torch.set_default_dtype(dt)
    try:
        yield
    finally:
        torch.set_default_dtype(prev)


@contextmanager
def default_device(dev: torch.device) -> Iterator[None]:
    prev = torch.tensor([]).device
    torch.set_default_device(dev)
    try:
        yield
    finally:
        torch.set_default_device(prev)


# ───────────────────────────── Tensor serialisation ──────────────────────────
class TensorState(BaseModel):
    """CPU‑only SafeTensors snapshot."""

    data: bytes
    shape: Tuple[int, ...]
    dtype: dtype

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=False)

    # .................................................................. #
    # Converters
    # .................................................................. #

    @staticmethod
    def from_torch(t: torch.Tensor) -> TensorState:
        if t.device != device.cpu.to_torch():
            raise RuntimeError("TensorState expects a CPU tensor.")
        return TensorState(
            data=_sf_save({"tensor": t}),
            shape=tuple(t.shape),
            dtype=dtype.from_torch(t.dtype),
        )

    def to_torch(self) -> torch.Tensor:
        with safe_open(
            BytesIO(self.data), framework="pt", device=device.cpu.value
        ) as f:
            tensor = f.get_tensor("tensor")
        if (
            tuple(tensor.shape) != self.shape
            or dtype.from_torch(tensor.dtype) != self.dtype
        ):
            raise RuntimeError("Stored metadata does not match tensor contents.")
        return tensor

    @staticmethod
    def from_bytes(raw: bytes) -> TensorState:
        with safe_open(BytesIO(raw), framework="pt", device=device.cpu.value) as f:
            keys = list(f.keys())
            if keys != ["tensor"]:
                raise ValueError(
                    "SafeTensor must contain a single entry named 'tensor'."
                )
            t = f.get_tensor("tensor")
        return TensorState(
            data=raw,
            shape=tuple(t.shape),
            dtype=dtype.from_torch(t.dtype),
        )


# ───────────────────────────── runtime fingerprint ───────────────────────────
class TorchEnv(BaseModel):
    torch_version: str
    cuda_version: str
    cudnn_version: int
    gpu_name: str
    python_version: str

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=False)

    @classmethod
    def snapshot(cls) -> TorchEnv:
        return cls(
            torch_version=torch.__version__,
            cuda_version=torch.version.cuda,
            cudnn_version=torch.backends.cudnn.version(),
            gpu_name=torch.cuda.get_device_name(0),
            python_version=platform.python_version(),
        )


# ───────────────────────────── Adam helpers (CPU‑only) ───────────────────────
class _HasStateDict(Protocol):
    def state_dict(self) -> Mapping[str, object]: ...


class AdamParamState(BaseModel):
    step: int
    exp_avg: TensorState
    exp_avg_sq: TensorState
    max_exp_avg_sq: TensorState | None

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=False)

    @classmethod
    def from_torch(cls, s: Mapping[str, object]) -> AdamParamState:
        # Ensure all tensors live on CPU
        if any(
            isinstance(v, torch.Tensor) and v.device != device.cpu.to_torch()
            for v in s.values()
        ):
            raise RuntimeError("All Adam state tensors must reside on CPU.")

        # Allow only the four known keys (max_exp_avg_sq is optional)
        unexpected = set(s) - {"step", "exp_avg", "exp_avg_sq", "max_exp_avg_sq"}
        assert not unexpected, f"Unexpected key(s) in Adam state: {unexpected}"

        return cls(
            step=int(s["step"]),
            exp_avg=TensorState.from_torch(s["exp_avg"]),
            exp_avg_sq=TensorState.from_torch(s["exp_avg_sq"]),
            max_exp_avg_sq=(
                TensorState.from_torch(s["max_exp_avg_sq"])
                if s.get("max_exp_avg_sq") is not None
                else None
            ),
        )

    def to_torch(self) -> Dict[str, object]:
        base = {
            "step": self.step,
            "exp_avg": self.exp_avg.to_torch(),
            "exp_avg_sq": self.exp_avg_sq.to_torch(),
        }
        return (
            base
            if self.max_exp_avg_sq is None
            else {**base, "max_exp_avg_sq": self.max_exp_avg_sq.to_torch()}
        )


class AdamParamGroup(BaseModel):
    params: List[int]
    lr: float
    betas: Tuple[float, float]
    eps: float
    weight_decay: float
    amsgrad: bool = False
    maximize: bool = False

    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=False)

    @classmethod
    def from_torch(cls, g: Mapping[str, object]) -> AdamParamGroup:
        return cls.model_validate(g)

    def to_torch(self) -> Dict[str, object]:
        return self.model_dump(mode="python")


class AdamOptimizerState(BaseModel):
    param_states: Dict[int, AdamParamState]
    param_groups: List[AdamParamGroup]

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=False)

    # .................................................................. #

    @classmethod
    def from_torch(cls, optim: _HasStateDict) -> AdamOptimizerState:
        if optim.__class__.__name__ not in {"Adam", "AdamW"}:
            raise TypeError("Only Adam or AdamW are supported.")

        sd = optim.state_dict()

        # ensure *exactly* the expected keys are present
        assert set(sd) == {"state", "param_groups"}, (
            f"optimizer.state_dict() must contain only 'state' and 'param_groups' "
            f"keys; got {set(sd)}"
        )
        ps = {
            pid: AdamParamState.from_torch(state) for pid, state in sd["state"].items()
        }
        groups = [AdamParamGroup.from_torch(pg) for pg in sd["param_groups"]]
        return cls(param_states=ps, param_groups=groups)

    # .................................................................. #

    def to_torch(self) -> Dict[str, object]:
        return {
            "state": {pid: st.to_torch() for pid, st in self.param_states.items()},
            "param_groups": [g.to_torch() for g in self.param_groups],
        }
