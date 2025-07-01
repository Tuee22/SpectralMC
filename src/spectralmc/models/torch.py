# src/spectralmc/models/torch.py
from __future__ import annotations

from contextlib import contextmanager
from enum import Enum
from typing import Dict, List, Mapping, Tuple, Iterator, Protocol

import io
import contextlib
import torch
import torch.nn as nn
from pydantic import BaseModel, ConfigDict
from safetensors import safe_open
from safetensors.torch import save as _sf_save

from .numerical import Precision

__all__ = [
    "dtype",
    "device",
    "TensorState",
    "ModelState",
    "LRSchedulerState",
    "AdamParamState",
    "AdamParamGroup",
    "AdamOptimizerState",
]

# ──────────────────────────────────────────────────────────────────────────────
# Dtype infrastructure
# ──────────────────────────────────────────────────────────────────────────────

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


class dtype(str, Enum):  # pylint: disable=invalid-name
    """Canonical identifiers for *all* supported Torch dtypes."""

    float16 = "float16"
    float32 = "float32"
    float64 = "float64"
    bfloat16 = "bfloat16"
    complex64 = "complex64"
    complex128 = "complex128"

    # ------------------------------------------------------------------ #
    # Torch-interop helpers
    # ------------------------------------------------------------------ #

    def to_torch(self) -> torch.dtype:
        return _DTYPE_STR_TO_TORCH[self.value]

    @classmethod
    def from_torch(cls, dt: torch.dtype) -> dtype:
        try:
            return cls(_TORCH_DTYPE_TO_STR[dt])
        except KeyError as exc:
            raise ValueError(f"Unsupported torch.dtype {dt!r}.") from exc

    # ------------------------------------------------------------------ #
    # Precision helpers
    # ------------------------------------------------------------------ #

    def to_precision(self) -> Precision:
        if self in (dtype.float32, dtype.float64):
            return Precision(self.value)
        raise ValueError(f"{self.value!r} cannot be represented as Precision.")

    @classmethod
    def from_precision(cls, p: Precision) -> dtype:
        return cls(p.value)


# ──────────────────────────────────────────────────────────────────────────────
# Device infrastructure
# ──────────────────────────────────────────────────────────────────────────────


class device(str, Enum):  # pylint: disable=invalid-name
    """High-level identifiers for common Torch devices."""

    cpu = "cpu"
    cuda = "cuda"

    def to_torch(self) -> torch.device:
        return torch.device(self.value)

    @classmethod
    def from_torch(cls, dev: torch.device) -> device:
        return cls(dev.type)


# ──────────────────────────────────────────────────────────────────────────────
# Context managers
# ──────────────────────────────────────────────────────────────────────────────


@contextmanager
def _default_dtype(dt: torch.dtype) -> Iterator[None]:
    prev = torch.get_default_dtype()
    torch.set_default_dtype(dt)
    try:
        yield
    finally:
        torch.set_default_dtype(prev)


@contextmanager
def _default_device(dev: torch.device) -> Iterator[None]:
    prev_dev = torch.tensor([]).device
    torch.set_default_device(dev)
    try:
        yield
    finally:
        torch.set_default_device(prev_dev)


# ──────────────────────────────────────────────────────────────────────────────
# Tensor snapshot (SafeTensors-backed)
# ──────────────────────────────────────────────────────────────────────────────


class TensorState(BaseModel):
    """SafeTensors-powered representation of a numeric `torch.Tensor`."""

    data: bytes
    shape: Tuple[int, ...]
    dtype: dtype

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=False)

    # ------------------------------------------------------------------ #
    # Converters
    # ------------------------------------------------------------------ #

    @staticmethod
    def from_tensor(tensor: torch.Tensor) -> TensorState:
        t_cpu = tensor.detach().cpu()
        return TensorState(
            data=_sf_save({"tensor": t_cpu}),
            shape=tuple(t_cpu.shape),
            dtype=dtype.from_torch(t_cpu.dtype),
        )

    def to_tensor(self, *, device: torch.device) -> torch.Tensor:
        buf = io.BytesIO(self.data)
        with contextlib.closing(
            safe_open(buf, framework="pt", device=str(device))
        ) as sf:
            return sf.get_tensor("tensor")


# ──────────────────────────────────────────────────────────────────────────────
# Protocol helpers
# ──────────────────────────────────────────────────────────────────────────────


class _HasStateDict(Protocol):
    def state_dict(self) -> Mapping[str, object]: ...


# ──────────────────────────────────────────────────────────────────────────────
# nn.Module snapshot
# ──────────────────────────────────────────────────────────────────────────────


class ModelState(BaseModel):
    """Captures parameters & buffers – architecture is out of scope."""

    parameters: Dict[str, TensorState]
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=False)

    @classmethod
    def from_torch(cls, model: nn.Module) -> ModelState:
        return cls(
            parameters={
                name: TensorState.from_tensor(tensor)
                for name, tensor in model.state_dict().items()
            }
        )

    def to_torch(self, *, device: torch.device) -> Dict[str, torch.Tensor]:
        return {k: ts.to_tensor(device=device) for k, ts in self.parameters.items()}


# ──────────────────────────────────────────────────────────────────────────────
# LR-scheduler snapshot
# ──────────────────────────────────────────────────────────────────────────────

_Number = int | float
_SchedulerValue = _Number | List[_Number]


class LRSchedulerState(BaseModel):
    """Serialised state of any standard LR scheduler."""

    state: Dict[str, _SchedulerValue]
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=False)

    @classmethod
    def from_torch(cls, scheduler: _HasStateDict) -> LRSchedulerState:
        raw = scheduler.state_dict()

        def parse(v: object) -> _SchedulerValue:
            if isinstance(v, (int, float)):
                return v
            if isinstance(v, list) and all(isinstance(x, (int, float)) for x in v):
                return [float(x) for x in v]
            raise TypeError(f"Unsupported value {type(v).__name__!r}")

        return cls(state={k: parse(v) for k, v in raw.items()})

    def to_torch(self) -> Dict[str, _SchedulerValue]:
        return dict(self.state)


# ──────────────────────────────────────────────────────────────────────────────
# Adam optimisation helpers
# ──────────────────────────────────────────────────────────────────────────────


class AdamParamState(BaseModel):
    """Per-parameter buffers for Adam / AdamW."""

    step: int
    exp_avg: TensorState
    exp_avg_sq: TensorState
    max_exp_avg_sq: TensorState | None = None

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=False)

    @classmethod
    def from_torch(cls, state: Mapping[str, object]) -> AdamParamState:
        to_int = lambda x: int(x.item() if isinstance(x, torch.Tensor) else x)

        mx = state.get("max_exp_avg_sq")
        return cls(
            step=to_int(state["step"]),
            exp_avg=TensorState.from_tensor(state["exp_avg"]),  # type: ignore[arg-type]
            exp_avg_sq=TensorState.from_tensor(state["exp_avg_sq"]),  # type: ignore[arg-type]
            max_exp_avg_sq=(
                None if mx is None else TensorState.from_tensor(mx)
            ),  # type: ignore[arg-type]
        )


class AdamParamGroup(BaseModel):
    """One entry from `optimizer.state_dict()['param_groups']`."""

    params: List[int]
    lr: float
    betas: Tuple[float, float]
    eps: float
    weight_decay: float
    amsgrad: bool = False
    maximize: bool = False

    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=False)

    @classmethod
    def from_torch(cls, group: Mapping[str, object]) -> AdamParamGroup:
        return cls.model_validate(group)

    def to_torch(self) -> Dict[str, object]:
        return self.model_dump(mode="python")


# --------------------------------------------------------------------------- #
# SafeTensors-powered Adam snapshot
# --------------------------------------------------------------------------- #


class AdamOptimizerState(BaseModel):
    """
    Snapshot of an Adam / AdamW optimiser, storing per-parameter tensors
    in **SafeTensors** for speed & safety.

    * `tensors` – raw bytes from `safetensors.torch.save`.
      Keys follow ``"{pid}/{buffer_name}"`` where
      ``buffer_name`` ∈ {"step", "exp_avg", "exp_avg_sq", "max_exp_avg_sq"}.
    * `param_groups` – the usual non-tensor hyper-parameters.
    """

    tensors: bytes
    param_groups: List[AdamParamGroup]

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=False)

    # .................................................................. #
    # Converters
    # .................................................................. #

    @classmethod
    def from_torch(cls, optimiser: _HasStateDict) -> AdamOptimizerState:
        if optimiser.__class__.__name__ not in {"Adam", "AdamW"}:
            raise TypeError("Only Adam or AdamW are supported.")

        sd = optimiser.state_dict()
        state, groups = sd["state"], sd["param_groups"]

        to_int = lambda x: int(x.item() if isinstance(x, torch.Tensor) else x)

        tensor_map: Dict[str, torch.Tensor] = {
            **{
                f"{pid}/step": torch.tensor(to_int(bufs["step"]), dtype=torch.float32)
                for pid, bufs in state.items()
            },
            **{
                f"{pid}/{key}": t.detach().cpu()
                for pid, bufs in state.items()
                for key, t in bufs.items()
                if key in {"exp_avg", "exp_avg_sq", "max_exp_avg_sq"}
                and isinstance(t, torch.Tensor)
            },
        }

        return cls(
            tensors=_sf_save(tensor_map),
            param_groups=[AdamParamGroup.from_torch(pg) for pg in groups],  # type: ignore[arg-type]
        )

    # .................................................................. #

    def to_torch(self, *, device: torch.device) -> Dict[str, object]:
        buf = io.BytesIO(self.tensors)
        with contextlib.closing(safe_open(buf, framework="pt")) as f:
            tensor_map = {name: f.get_tensor(name) for name in f.keys()}

        state_out: Dict[int, Dict[str, object]] = {
            int(pid): (
                {"step": t.to("cpu")} if field == "step" else {field: t.to(device)}
            )
            for pid_field, t in tensor_map.items()
            for pid, field in [pid_field.split("/", 1)]
        }

        # Merge entries that belong to the same pid
        merged_state: Dict[int, Dict[str, object]] = {}
        for pid, kv in state_out.items():
            merged_state.setdefault(pid, {}).update(kv)

        return {
            "state": merged_state,
            "param_groups": [pg.to_torch() for pg in self.param_groups],
        }
