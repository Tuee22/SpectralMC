# src/spectralmc/models/torch.py
from __future__ import annotations

"""
Loss-less, version-robust, **mypy-strict** helpers for serialising and
deserialising common PyTorch objects:

* **Tensor →** :class:`TensorState`
* **nn.Module →** :class:`ModelState`
* **LR scheduler →** :class:`LRSchedulerState`
* **Adam / AdamW optimiser →** :class:`AdamOptimizerState` (with
  :class:`AdamParamState` & :class:`AdamParamGroup`)

Design goals
------------

1. **Round-trip safety** – Objects produced via ``*.to_torch`` load cleanly
   through the corresponding ``load_state_dict`` **today** and in future
   PyTorch releases.

2. **Version robustness** – Historical quirks (e.g. Adam’s ``step`` changing
   from *int* ➜ rank-0 CPU tensor in v2.3) are normalised automatically.

3. **Strict typing** – The file passes ``mypy --strict`` with *zero*
   ``Any``, ``cast`` or ``# type: ignore`` directives.

Everything intended for external use is re-exported via :pydata:`__all__`.
"""

from typing import Dict, List, Mapping, MutableMapping, Optional, Tuple, Protocol

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pydantic import BaseModel, ConfigDict

__all__ = [
    "TensorState",
    "ModelState",
    "LRSchedulerState",
    "AdamParamState",
    "AdamParamGroup",
    "AdamOptimizerState",
]

# ──────────────────────────────────────────────────────────────────────────────
# Helper maps & utilities
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


def _torch_dtype_from_str(name: str) -> torch.dtype:
    """Convert a canonical dtype *name* → :class:`torch.dtype`."""
    if name not in _DTYPE_STR_TO_TORCH:
        raise ValueError(f"Unsupported dtype string {name!r}.")
    return _DTYPE_STR_TO_TORCH[name]


def _torch_dtype_to_str(dtype: torch.dtype) -> str:
    """Convert :class:`torch.dtype` → its canonical string name."""
    if dtype not in _TORCH_DTYPE_TO_STR:
        raise ValueError(f"Unsupported torch.dtype {dtype!r}.")
    return _TORCH_DTYPE_TO_STR[dtype]


def _coerce_to_int(value: object) -> int:
    """
    Safely convert *value* to :class:`int`.

    Accepts Python ``int``, NumPy integral scalars and rank-0 tensors.
    """
    if isinstance(value, int):
        return value
    if isinstance(value, torch.Tensor) and value.ndim == 0:
        return int(value.item())
    if isinstance(value, np.integer):
        return int(value)
    raise TypeError(f"Expected int-like value, got {type(value).__name__}.")


# ──────────────────────────────────────────────────────────────────────────────
# Generic tensor snapshot
# ──────────────────────────────────────────────────────────────────────────────


class TensorState(BaseModel):
    """
    JSON-friendly representation of a :class:`torch.Tensor`.

    Only numeric tensors are supported; the snapshot detaches data to *CPU* and
    stores a flattened ``list`` plus shape & dtype.
    """

    data: List[float]
    shape: Tuple[int, ...]
    dtype: str

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=False)

    # --------------------------------------------------------------------- #
    # Converters
    # --------------------------------------------------------------------- #

    @staticmethod
    def from_tensor(tensor: torch.Tensor) -> TensorState:
        """
        Capture *tensor*.

        Parameters
        ----------
        tensor:
            The tensor to snapshot.

        Returns
        -------
        TensorState
        """
        t_cpu = tensor.detach().cpu()
        return TensorState(
            data=t_cpu.reshape(-1).tolist(),
            shape=tuple(t_cpu.shape),
            dtype=_torch_dtype_to_str(t_cpu.dtype),
        )

    def to_tensor(self, *, device: torch.device | str = "cpu") -> torch.Tensor:
        """
        Materialise the original tensor.

        Parameters
        ----------
        device:
            Destination device (default ``"cpu"``).

        Returns
        -------
        torch.Tensor
        """
        return (
            torch.tensor(
                self.data,
                dtype=_torch_dtype_from_str(self.dtype),
                device=device,
            )
            .reshape(self.shape)
            .clone()
        )


# ──────────────────────────────────────────────────────────────────────────────
# Protocol helpers (avoid “untyped call” mypy errors)
# ──────────────────────────────────────────────────────────────────────────────


class _HasStateDict(Protocol):
    """Minimal protocol – any object exposing *typed* ``state_dict``."""

    def state_dict(self) -> Mapping[str, object]:  # noqa: D401 – simple signature
        ...


# ──────────────────────────────────────────────────────────────────────────────
# nn.Module wrapper
# ──────────────────────────────────────────────────────────────────────────────


class ModelState(BaseModel):
    """
    Serialised weights of an :class:`torch.nn.Module`.

    **Note** – Only *parameters & buffers* are captured. Architecture and
    hyper-parameters are outside the scope of this snapshot.
    """

    parameters: Dict[str, TensorState]

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=False)

    # --------------------------------------------------------------------- #
    # Converters
    # --------------------------------------------------------------------- #

    @classmethod
    def from_torch(cls, model: nn.Module) -> "ModelState":
        """
        Capture *model*.

        Parameters
        ----------
        model:
            An initialised :class:`torch.nn.Module`.

        Returns
        -------
        ModelState
        """
        state_dict_raw = model.state_dict()
        param_state = {
            name: TensorState.from_tensor(tensor)
            for name, tensor in state_dict_raw.items()
        }
        return cls(parameters=param_state)

    def to_torch(
        self, *, device: torch.device | str = "cpu"
    ) -> Dict[str, torch.Tensor]:
        """
        Build a PyTorch-compatible *state-dict*.

        Parameters
        ----------
        device:
            Target device for tensors.

        Returns
        -------
        dict
        """
        return {
            name: ts.to_tensor(device=device) for name, ts in self.parameters.items()
        }


# ──────────────────────────────────────────────────────────────────────────────
# LR-scheduler wrapper
# ──────────────────────────────────────────────────────────────────────────────

_Number = int | float
_SchedulerValue = _Number | List[_Number]


class LRSchedulerState(BaseModel):
    """
    Serialised state of a :class:`torch.optim.lr_scheduler._LRScheduler`.

    The snapshot does *not* include the scheduler *type*; users must
    reconstruct an identical scheduler instance before loading.
    """

    state: Dict[str, _SchedulerValue]

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=False)

    # --------------------------------------------------------------------- #
    # Converters
    # --------------------------------------------------------------------- #

    @classmethod
    def from_torch(cls, scheduler: _HasStateDict) -> "LRSchedulerState":
        """
        Capture *scheduler*.

        Parameters
        ----------
        scheduler:
            Any object exposing a ``state_dict`` whose values are numbers or
            lists of numbers (compatible with standard LR schedulers).

        Returns
        -------
        LRSchedulerState
        """
        raw_state = scheduler.state_dict()

        parsed: Dict[str, _SchedulerValue] = {}
        for key, val in raw_state.items():
            if isinstance(val, (int, float)):
                parsed[key] = val
            elif isinstance(val, list) and all(
                isinstance(x, (int, float)) for x in val
            ):
                # Re-box into List[float] to satisfy mypy’s strictness.
                parsed[key] = [float(x) for x in val]
            else:
                raise TypeError(
                    f"Unsupported value type {type(val).__name__!r} for key {key!r}."
                )
        return cls(state=parsed)

    def to_torch(self) -> Dict[str, _SchedulerValue]:
        """Return the raw mapping ready for ``load_state_dict``."""
        return dict(self.state)


# ──────────────────────────────────────────────────────────────────────────────
# Adam-specific helpers
# ──────────────────────────────────────────────────────────────────────────────


class AdamParamState(BaseModel):
    """
    Per-parameter buffers for Adam / AdamW.

    Older layouts are accepted transparently by :meth:`from_torch`; *export* is
    always in the current canonical format (rank-0 CPU tensor for ``step``).
    """

    step: int
    exp_avg: TensorState
    exp_avg_sq: TensorState
    max_exp_avg_sq: Optional[TensorState] = None

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=False)

    # --------------------------------------------------------------------- #
    # Converters
    # --------------------------------------------------------------------- #

    @classmethod
    def from_torch(cls, state: Mapping[str, object]) -> "AdamParamState":
        """
        Convert a single *per-parameter* mapping → :class:`AdamParamState`.
        """
        step_int = _coerce_to_int(state["step"])

        exp_avg_t = state["exp_avg"]
        exp_avg_sq_t = state["exp_avg_sq"]

        if not isinstance(exp_avg_t, torch.Tensor) or not isinstance(
            exp_avg_sq_t, torch.Tensor
        ):
            raise TypeError("Both 'exp_avg' and 'exp_avg_sq' must be tensors.")

        max_state: Optional[TensorState]
        if state.get("max_exp_avg_sq") is not None:
            max_t = state["max_exp_avg_sq"]
            if not isinstance(max_t, torch.Tensor):
                raise TypeError("'max_exp_avg_sq' must be a tensor if present.")
            max_state = TensorState.from_tensor(max_t)
        else:
            max_state = None

        return cls(
            step=step_int,
            exp_avg=TensorState.from_tensor(exp_avg_t),
            exp_avg_sq=TensorState.from_tensor(exp_avg_sq_t),
            max_exp_avg_sq=max_state,
        )

    def to_torch(self, *, device: torch.device | str = "cpu") -> Dict[str, object]:
        """
        Emit a mapping compatible with ``optimizer.load_state_dict``.

        ``step`` is saved as a rank-0 *CPU* tensor (float32) per PyTorch 2.3+.
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
    """
    One entry from ``optimizer.state_dict()['param_groups']`` (Adam / AdamW).
    """

    params: List[int]
    lr: float
    betas: Tuple[float, float]
    eps: float
    weight_decay: float
    amsgrad: bool = False
    maximize: bool = False

    # Rare extra keys (capturable, foreach, …) are preserved verbatim to avoid
    # lossy round-trips.
    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=False)

    # --------------------------------------------------------------------- #
    # Converters
    # --------------------------------------------------------------------- #

    @classmethod
    def from_torch(cls, group: Mapping[str, object]) -> "AdamParamGroup":
        """Validate *group* against the model schema."""
        return cls.model_validate(group)

    def to_torch(self) -> Dict[str, object]:
        """Return the mapping in PyTorch-native format."""
        return self.model_dump(mode="python")


class AdamOptimizerState(BaseModel):
    """
    Full snapshot of an :class:`~torch.optim.Adam` / :class:`~torch.optim.AdamW`.
    """

    state: Dict[int, AdamParamState]
    param_groups: List[AdamParamGroup]

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=False)

    # --------------------------------------------------------------------- #
    # Converters
    # --------------------------------------------------------------------- #

    @classmethod
    def from_torch(cls, optimizer: _HasStateDict) -> AdamOptimizerState:
        """
        Capture *optimizer*.

        Parameters
        ----------
        optimizer:
            Instance of Adam / AdamW. Runtime validation is performed using the
            class name to avoid stub-level issues.

        Returns
        -------
        AdamOptimizerState
        """
        if optimizer.__class__.__name__ not in {"Adam", "AdamW"}:
            raise TypeError("Only Adam or AdamW optimisers are supported.")

        raw_sd = optimizer.state_dict()  # Mapping[str, object]
        raw_state = raw_sd["state"]
        raw_groups = raw_sd["param_groups"]

        if not isinstance(raw_state, dict):
            raise TypeError("'state' must be a dict.")
        if not isinstance(raw_groups, list):
            raise TypeError("'param_groups' must be a list.")

        state_conv: Dict[int, AdamParamState] = {}
        for pid, bufs in raw_state.items():
            if not isinstance(pid, int) or not isinstance(bufs, Mapping):
                raise TypeError("Unexpected key or value type in 'state' mapping.")
            state_conv[pid] = AdamParamState.from_torch(bufs)

        groups_conv = [AdamParamGroup.from_torch(pg) for pg in raw_groups]

        return cls(state=state_conv, param_groups=groups_conv)

    def to_torch(self, *, device: torch.device | str = "cpu") -> Dict[str, object]:
        """
        Build a *state-dict* suitable for ``optimizer.load_state_dict``.
        """
        state_out = {pid: st.to_torch(device=device) for pid, st in self.state.items()}
        groups_out = [pg.to_torch() for pg in self.param_groups]
        return {"state": state_out, "param_groups": groups_out}
