# src/spectralmc/models/torch.py
from __future__ import annotations

"""
Loss-less, version-robust, **mypy-clean** helpers to serialise the complete
state of :class:`torch.optim.Adam` / :class:`torch.optim.AdamW`.

Why this module exists
----------------------
* PyTorch stores optimiser buffers in a *large* nested ``dict`` containing both
  Python scalars and tensors.  This format is not JSON-serialisable and is
  brittle across PyTorch versions.
* Starting with **PyTorch 2.3** the per-parameter field ``"step"`` changed from
  a Python ``int`` ➜ a rank-0 **CPU** tensor.  If that tensor is moved to the
  GPU the *multi-tensor* Adam implementation aborts with

      RuntimeError: Tensors of the same index must be on the same device …

  Therefore we **must**:
    1. Accept both the old *int* and the new *tensor* form when *reading* a
       state-dict.
    2. Always write the *tensor* form **on the CPU** when *exporting* so that a
       round-trip can be loaded by current PyTorch without error.

The classes below meet these requirements while remaining strictly typed:
there is **no** ``Any``, ``cast`` or ``# type: ignore`` anywhere and
``mypy --strict`` passes cleanly.
"""

from typing import Dict, List, Mapping, Optional, Tuple, Union

import math

import numpy as np
import torch
from pydantic import BaseModel, ConfigDict

__all__ = ["AdamOptimizerState"]

# --------------------------------------------------------------------------- #
# Helper maps                                                                  #
# --------------------------------------------------------------------------- #

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


def _torch_dtype_from_str(name: str) -> torch.dtype:
    """Convert canonical dtype *name* → :class:`torch.dtype`."""
    if name not in _DTYPE_STR_TO_TORCH:
        raise ValueError(f"Unsupported dtype string {name!r}.")
    return _DTYPE_STR_TO_TORCH[name]


def _torch_dtype_to_str(dtype: torch.dtype) -> str:
    """Convert :class:`torch.dtype` → canonical string name."""
    if dtype not in _TORCH_DTYPE_TO_STR:
        raise ValueError(f"Unsupported torch.dtype {dtype!r}.")
    return _TORCH_DTYPE_TO_STR[dtype]


def _coerce_to_int(value: object) -> int:
    """Return *value* as ``int`` if it is scalar-integer-like."""
    if isinstance(value, int):
        return value
    if isinstance(value, torch.Tensor) and value.ndim == 0:
        return int(value.item())
    if isinstance(value, np.integer):
        return int(value)
    raise TypeError(f"Expected int-like value, got {type(value).__name__}.")


# --------------------------------------------------------------------------- #
# Pydantic models                                                              #
# --------------------------------------------------------------------------- #


class AdamTensorState(BaseModel):
    """JSON-friendly snapshot of a :class:`torch.Tensor`."""

    data: List[float]
    shape: Tuple[int, ...]
    dtype: str

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=False)

    # Validators ----------------------------------------------------------- #
    def model_post_init(self, __context: object) -> None:  # noqa: D401
        if len(self.data) != math.prod(self.shape):
            raise ValueError("Tensor data length does not match shape.")

    # Converters ----------------------------------------------------------- #
    @staticmethod
    def from_tensor(tensor: torch.Tensor) -> "AdamTensorState":
        """Detaches *tensor* to CPU and creates a serialisable view."""
        t_cpu = tensor.detach().cpu()
        return AdamTensorState(
            data=t_cpu.reshape(-1).tolist(),
            shape=tuple(t_cpu.shape),
            dtype=_torch_dtype_to_str(t_cpu.dtype),
        )

    def to_tensor(self, *, device: torch.device | str = "cpu") -> torch.Tensor:
        """Rebuild the original tensor on *device*."""
        return torch.tensor(
            self.data,
            dtype=_torch_dtype_from_str(self.dtype),
            device=device,
        ).reshape(self.shape)


class AdamParamState(BaseModel):
    """Per-parameter buffers for Adam / AdamW."""

    step: int
    exp_avg: AdamTensorState
    exp_avg_sq: AdamTensorState
    max_exp_avg_sq: Optional[AdamTensorState] = None

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    # Converters ----------------------------------------------------------- #
    @classmethod
    def from_torch(
        cls, state: Mapping[str, Union[int, torch.Tensor, None]]
    ) -> "AdamParamState":
        """Convert a raw PyTorch mapping → :class:`AdamParamState`."""
        step_int = _coerce_to_int(state["step"])

        exp_avg_t = state["exp_avg"]
        exp_avg_sq_t = state["exp_avg_sq"]
        if not isinstance(exp_avg_t, torch.Tensor) or not isinstance(
            exp_avg_sq_t, torch.Tensor
        ):
            raise TypeError("exp_avg and exp_avg_sq must be tensors.")

        if "max_exp_avg_sq" in state and state["max_exp_avg_sq"] is not None:
            max_t = state["max_exp_avg_sq"]
            if not isinstance(max_t, torch.Tensor):
                raise TypeError("max_exp_avg_sq must be a tensor if present.")
            max_state: Optional[AdamTensorState] = AdamTensorState.from_tensor(max_t)
        else:
            max_state = None

        return cls(
            step=step_int,
            exp_avg=AdamTensorState.from_tensor(exp_avg_t),
            exp_avg_sq=AdamTensorState.from_tensor(exp_avg_sq_t),
            max_exp_avg_sq=max_state,
        )

    def to_torch(self, *, device: torch.device | str = "cpu") -> Dict[str, object]:
        """Return a mapping compatible with ``optimizer.load_state_dict``.

        Notes
        -----
        * ``step`` is emitted as a **CPU float32 tensor** to mirror the format
          used by modern PyTorch.  Keeping it on CPU avoids the *same-device*
          restriction enforced by the multi-tensor Adam code path.
        """
        out: Dict[str, object] = {
            "step": torch.tensor(self.step, dtype=torch.float32, device="cpu"),
            "exp_avg": self.exp_avg.to_tensor(device=device),
            "exp_avg_sq": self.exp_avg_sq.to_tensor(device=device),
        }
        if self.max_exp_avg_sq is not None:
            out["max_exp_avg_sq"] = self.max_exp_avg_sq.to_tensor(device=device)
        return out


class AdamParamGroup(BaseModel):
    """One entry in ``optimizer.state_dict()['param_groups']``."""

    params: List[int]
    lr: float
    betas: Tuple[float, float]
    eps: float
    weight_decay: float
    amsgrad: bool = False
    maximize: bool = False

    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)

    # Converters ----------------------------------------------------------- #
    @classmethod
    def from_torch(cls, group: Mapping[str, object]) -> "AdamParamGroup":
        return cls.model_validate(group)

    def to_torch(self) -> Dict[str, object]:
        return self.model_dump(mode="python")


class AdamOptimizerState(BaseModel):
    """Full snapshot of an Adam / AdamW optimiser."""

    state: Dict[int, AdamParamState]
    param_groups: List[AdamParamGroup]

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    # Converters ----------------------------------------------------------- #
    @classmethod
    def from_torch(cls, sd: Mapping[str, object]) -> "AdamOptimizerState":
        if "state" not in sd or "param_groups" not in sd:
            raise KeyError("state_dict must contain 'state' and 'param_groups'.")

        raw_state = sd["state"]
        raw_groups = sd["param_groups"]

        if not isinstance(raw_state, dict):
            raise TypeError("'state' must be a dict.")
        if not isinstance(raw_groups, list):
            raise TypeError("'param_groups' must be a list.")

        state_conv = {
            pid: AdamParamState.from_torch(bufs)
            for pid, bufs in raw_state.items()
            if isinstance(pid, int) and isinstance(bufs, Mapping)
        }
        if len(state_conv) != len(raw_state):
            raise TypeError("Unexpected key or value type in 'state' mapping.")

        groups_conv = [AdamParamGroup.from_torch(pg) for pg in raw_groups]

        return cls(state=state_conv, param_groups=groups_conv)

    def to_torch(self, *, device: torch.device | str = "cpu") -> Dict[str, object]:
        state_out = {pid: st.to_torch(device=device) for pid, st in self.state.items()}
        groups_out = [pg.to_torch() for pg in self.param_groups]
        return {"state": state_out, "param_groups": groups_out}
