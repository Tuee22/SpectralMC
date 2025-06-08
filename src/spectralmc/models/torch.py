from __future__ import annotations

"""
Torch-compatible optimizer serialization helpers for Adam.

Includes:
    - AdamTensorState: Serializable torch.Tensor snapshot
    - AdamParamState: Per-parameter optimizer state
    - AdamParamGroup: Hyperparameter group
    - AdamOptimizerState: Full optimizer snapshot
"""

import math
from typing import Dict, List, Mapping, Optional, Tuple

import torch
from pydantic import BaseModel, ConfigDict

__all__ = [
    "AdamTensorState",
    "AdamParamState",
    "AdamParamGroup",
    "AdamOptimizerState",
]


def _torch_dtype_from_str(name: str) -> torch.dtype:
    mapping: Dict[str, torch.dtype] = {
        "float16": torch.float16,
        "float32": torch.float32,
        "float64": torch.float64,
        "bfloat16": torch.bfloat16,
        "complex64": torch.complex64,
        "complex128": torch.complex128,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported dtype string '{name}'.")
    return mapping[name]


def _torch_dtype_to_str(dtype: torch.dtype) -> str:
    reverse: Dict[torch.dtype, str] = {
        torch.float16: "float16",
        torch.float32: "float32",
        torch.float64: "float64",
        torch.bfloat16: "bfloat16",
        torch.complex64: "complex64",
        torch.complex128: "complex128",
    }
    if dtype not in reverse:
        raise ValueError(f"Unsupported torch.dtype '{dtype}'.")
    return reverse[dtype]


class AdamTensorState(BaseModel):
    data: List[float]
    shape: Tuple[int, ...]
    dtype: str

    def model_post_init(self, __context: object) -> None:
        if len(self.data) != math.prod(self.shape):
            raise ValueError("Tensor data length does not match shape.")

    @staticmethod
    def from_tensor(tensor: torch.Tensor) -> AdamTensorState:
        t_cpu = tensor.detach().cpu()
        return AdamTensorState(
            data=t_cpu.reshape(-1).tolist(),
            shape=tuple(t_cpu.shape),
            dtype=_torch_dtype_to_str(t_cpu.dtype),
        )

    def to_tensor(self, *, device: torch.device | str = "cpu") -> torch.Tensor:
        return torch.tensor(
            self.data, dtype=_torch_dtype_from_str(self.dtype), device=device
        ).reshape(self.shape)


class AdamParamState(BaseModel):
    step: int
    exp_avg: AdamTensorState
    exp_avg_sq: AdamTensorState
    max_exp_avg_sq: Optional[AdamTensorState] = None

    @staticmethod
    def from_torch(state: Mapping[str, object]) -> AdamParamState:
        raw_step = state.get("step")
        if not isinstance(raw_step, int):
            raise TypeError(f"Expected int for 'step', got {type(raw_step).__name__}")

        return AdamParamState(
            step=raw_step,
            exp_avg=AdamTensorState.from_tensor(state["exp_avg"]),
            exp_avg_sq=AdamTensorState.from_tensor(state["exp_avg_sq"]),
            max_exp_avg_sq=(
                AdamTensorState.from_tensor(state["max_exp_avg_sq"])
                if "max_exp_avg_sq" in state and state["max_exp_avg_sq"] is not None
                else None
            ),
        )

    def to_torch(self, *, device: torch.device | str = "cpu") -> Dict[str, object]:
        out: Dict[str, object] = {
            "step": self.step,
            "exp_avg": self.exp_avg.to_tensor(device=device),
            "exp_avg_sq": self.exp_avg_sq.to_tensor(device=device),
        }
        if self.max_exp_avg_sq is not None:
            out["max_exp_avg_sq"] = self.max_exp_avg_sq.to_tensor(device=device)
        return out


class AdamParamGroup(BaseModel):
    params: List[int]
    lr: float
    betas: Tuple[float, float]
    eps: float
    weight_decay: float
    amsgrad: bool = False
    maximize: bool = False

    model_config = ConfigDict(extra="allow")

    @staticmethod
    def from_torch(group: Mapping[str, object]) -> AdamParamGroup:
        return AdamParamGroup.model_validate(group)

    def to_torch(self) -> Dict[str, object]:
        return self.model_dump(mode="python")


class AdamOptimizerState(BaseModel):
    state: Dict[int, AdamParamState]
    param_groups: List[AdamParamGroup]

    @staticmethod
    def from_torch(sd: Mapping[str, object]) -> AdamOptimizerState:
        raw_state = sd["state"]
        raw_groups = sd["param_groups"]

        assert isinstance(raw_state, dict)
        assert isinstance(raw_groups, list)

        return AdamOptimizerState(
            state={int(k): AdamParamState.from_torch(v) for k, v in raw_state.items()},
            param_groups=[AdamParamGroup.from_torch(pg) for pg in raw_groups],
        )

    def to_torch(self, *, device: torch.device | str = "cpu") -> Dict[str, object]:
        return {
            "state": {k: v.to_torch(device=device) for k, v in self.state.items()},
            "param_groups": [pg.to_torch() for pg in self.param_groups],
        }
