"""
GPU trainer that learns the discrete Fourier transform (DFT) of discounted
Black-Scholes pay-off distributions with a complex-valued neural network
(**CVNN**) entirely on CUDA.

Highlights
----------
* **CUDA-only training** – :pymeth:`train` raises if no CUDA device is present,
  but CPU inference is fully supported.
* **Dedicated streams** – CuPy FFTs and PyTorch autograd kernels run on
  separate CUDA streams.
* **Pluggable logging** – any callable matching :data:`StepLogger` can consume
  per-iteration metrics (see :class:`TensorBoardLogger`).
* **Deterministic snapshot / restore** – all mutable state, including the Adam
  optimiser buffers, lives in :class:`GbmTrainerConfig`.
"""

from __future__ import annotations

import math
import time
import warnings
from typing import Callable, Dict, List, Mapping, Optional, Tuple, TypeAlias

import cupy as cp
from cupy.cuda import Stream as CuPyStream
import torch
import torch.nn as nn
import torch.optim as optim
from pydantic import BaseModel, ConfigDict
from torch.utils.tensorboard import SummaryWriter

from spectralmc.cvnn import CVNN
from spectralmc.gbm import BlackScholes, BlackScholesConfig
from spectralmc.sobol_sampler import BoundSpec, SobolSampler

# --------------------------------------------------------------------------- #
# Typing helpers                                                              #
# --------------------------------------------------------------------------- #

StepLogger: TypeAlias = Callable[["StepMetrics"], None]

# --------------------------------------------------------------------------- #
# Adam-state serialisation helpers                                            #
# --------------------------------------------------------------------------- #


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
    """JSON-friendly snapshot of a PyTorch tensor."""

    data: List[float]
    shape: Tuple[int, ...]
    dtype: str

    # Pydantic v2 hook -------------------------------------------------- #
    def model_post_init(self, __context: object) -> None:  # noqa: D401
        """Validate *data* length versus :pyattr:`shape` product."""
        if len(self.data) != math.prod(self.shape):
            raise ValueError(
                "Flat `data` length does not match the product of `shape`."
            )

    # Conversions ------------------------------------------------------- #
    @staticmethod
    def from_tensor(tensor: torch.Tensor) -> "AdamTensorState":
        t_cpu = tensor.detach().cpu()
        return AdamTensorState(
            data=t_cpu.reshape(-1).tolist(),
            shape=tuple(t_cpu.shape),
            dtype=_torch_dtype_to_str(t_cpu.dtype),
        )

    def to_tensor(self, *, device: torch.device | str = "cpu") -> torch.Tensor:
        return torch.tensor(
            self.data,
            dtype=_torch_dtype_from_str(self.dtype),
            device=device,
        ).reshape(self.shape)


class AdamParamState(BaseModel):
    """Serialisable state for one Adam parameter slot."""

    step: int
    exp_avg: AdamTensorState
    exp_avg_sq: AdamTensorState
    max_exp_avg_sq: Optional[AdamTensorState] = None

    @staticmethod
    def from_torch(state: Mapping[str, object]) -> "AdamParamState":
        step = int(state["step"])
        exp_avg = AdamTensorState.from_tensor(state["exp_avg"])  # type: ignore[arg-type]
        exp_avg_sq = AdamTensorState.from_tensor(state["exp_avg_sq"])  # type: ignore[arg-type]
        max_exp: Optional[AdamTensorState]
        if "max_exp_avg_sq" in state and state["max_exp_avg_sq"] is not None:
            max_exp = AdamTensorState.from_tensor(state["max_exp_avg_sq"])  # type: ignore[arg-type]
        else:
            max_exp = None
        return AdamParamState(
            step=step, exp_avg=exp_avg, exp_avg_sq=exp_avg_sq, max_exp_avg_sq=max_exp
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
    """Hyper-parameter group of Adam (extra keys forwarded verbatim)."""

    params: List[int]
    lr: float
    betas: Tuple[float, float]
    eps: float
    weight_decay: float
    amsgrad: bool = False
    maximize: bool = False

    model_config = ConfigDict(extra="allow")

    @staticmethod
    def from_torch(group: Mapping[str, object]) -> "AdamParamGroup":
        # Let Pydantic validate & coerce – no **kwargs trickery needed.
        return AdamParamGroup.model_validate(group)

    def to_torch(self) -> Dict[str, object]:
        return self.model_dump(mode="python")


class AdamOptimizerState(BaseModel):
    """Fully typed, JSON-serialisable snapshot of an Adam optimiser."""

    state: Dict[int, AdamParamState]
    param_groups: List[AdamParamGroup]

    # Mapping convenience for legacy tests ----------------------------- #
    def __getitem__(self, key: str) -> object:  # noqa: D401
        if key == "state":
            return self.state
        if key == "param_groups":
            return self.param_groups
        raise KeyError(key)

    # Conversions ------------------------------------------------------- #
    @staticmethod
    def from_torch(sd: Mapping[str, object]) -> "AdamOptimizerState":
        raw_state = sd["state"]
        raw_groups = sd["param_groups"]
        assert isinstance(raw_state, dict) and isinstance(raw_groups, list)
        snap_state = {
            int(pid): AdamParamState.from_torch(pstate)  # type: ignore[arg-type]
            for pid, pstate in raw_state.items()
        }
        snap_groups = [AdamParamGroup.from_torch(pg) for pg in raw_groups]  # type: ignore[arg-type]
        return AdamOptimizerState(state=snap_state, param_groups=snap_groups)

    def to_torch(self, *, device: torch.device | str = "cpu") -> Dict[str, object]:
        return {
            "state": {
                pid: ps.to_torch(device=device) for pid, ps in self.state.items()
            },
            "param_groups": [pg.to_torch() for pg in self.param_groups],
        }


# --------------------------------------------------------------------------- #
# Core snapshot & metrics models                                              #
# --------------------------------------------------------------------------- #


class GbmTrainerConfig(BaseModel):
    """Deterministic snapshot of a :class:`GbmTrainer` instance."""

    cfg: BlackScholesConfig
    domain_bounds: Dict[str, BoundSpec]
    cvnn: CVNN
    optimizer_state: Optional[AdamOptimizerState] = None
    global_step: int = 0

    model_config = ConfigDict(arbitrary_types_allowed=True)


class StepMetrics(BaseModel):
    """Per-iteration metrics forwarded to a :data:`StepLogger`."""

    step: int
    batch_time: float
    loss: float
    grad_norm: float
    lr: float
    optimizer: optim.Adam
    optimizer_state: AdamOptimizerState
    model: CVNN

    model_config = ConfigDict(arbitrary_types_allowed=True)


# --------------------------------------------------------------------------- #
# TensorBoard logger                                                          #
# --------------------------------------------------------------------------- #


class TensorBoardLogger:
    """Write :class:`StepMetrics` streams to TensorBoard."""

    def __init__(
        self,
        *,
        logdir: str = ".logs/gbm_trainer",
        hist_every: int = 10,
        flush_every: int = 100,
    ) -> None:
        self._writer = SummaryWriter(log_dir=logdir)
        self._hist_every = max(1, hist_every)
        self._flush_every = max(1, flush_every)

    def __call__(self, metrics: StepMetrics) -> None:  # noqa: D401
        w = self._writer
        step = metrics.step

        w.add_scalar("Loss/train", metrics.loss, step)
        w.add_scalar("LR", metrics.lr, step)
        w.add_scalar("GradNorm", metrics.grad_norm, step)
        w.add_scalar("BatchTime", metrics.batch_time, step)

        if step % self._hist_every == 0:
            for name, param in metrics.model.named_parameters():
                w.add_histogram(name, param, step)
                if param.grad is not None:
                    w.add_histogram(f"{name}.grad", param.grad, step)

        if step % self._flush_every == 0:
            w.flush()

    def close(self) -> None:
        self._writer.close()


# --------------------------------------------------------------------------- #
# Utility helpers                                                             #
# --------------------------------------------------------------------------- #


def _torch_precision_dtype(precision: str) -> torch.dtype:
    return torch.float32 if precision == "float32" else torch.float64


def _split_inputs(
    inputs: List[BlackScholes.Inputs],
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    fields = list(BlackScholes.Inputs.model_fields.keys())
    rows = [[float(getattr(inp, f)) for f in fields] for inp in inputs]
    real = torch.tensor(rows, dtype=dtype, device=device)
    imag = torch.zeros_like(real)
    return real, imag


# --------------------------------------------------------------------------- #
# Trainer                                                                     #
# --------------------------------------------------------------------------- #


class GbmTrainer:
    """Coordinates MC pricing, FFT and CVNN optimisation."""

    # ------------------------------------------------------------------ #
    # Construction / snapshot                                            #
    # ------------------------------------------------------------------ #

    def __init__(self, cfg: GbmTrainerConfig) -> None:
        self._sim_params = cfg.cfg.sim_params
        self._cvnn = cfg.cvnn
        self._domain_bounds = cfg.domain_bounds
        self._optimizer_state = cfg.optimizer_state
        self._global_step = cfg.global_step

        # Device & streams --------------------------------------------- #
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self._device.type == "cuda":
            self._torch_stream: Optional[torch.cuda.Stream] = torch.cuda.Stream(
                device=self._device, priority=0
            )
            self._cupy_stream: Optional[CuPyStream] = CuPyStream(non_blocking=False)
        else:
            self._torch_stream = None
            self._cupy_stream = None

        # Sampler & MC engine ------------------------------------------ #
        self._sampler: SobolSampler[BlackScholes.Inputs] = SobolSampler(
            pydantic_class=BlackScholes.Inputs,
            dimensions=self._domain_bounds,
            skip=self._sim_params.skip,
            seed=self._sim_params.mc_seed,
        )
        self._mc_engine = BlackScholes(cfg.cfg)

        # Model device/dtype ------------------------------------------- #
        self._cvnn.to(self._device, _torch_precision_dtype(self._sim_params.dtype))

        # Cached constants --------------------------------------------- #
        self._network_size = self._sim_params.network_size
        self._batches_per_mc_run = self._sim_params.batches_per_mc_run
        self._cp_cdtype = (
            cp.complex64 if self._sim_params.dtype == "float32" else cp.complex128
        )
        self._torch_cdtype = (
            torch.complex64 if self._sim_params.dtype == "float32" else torch.complex128
        )
        self._torch_rdtype = _torch_precision_dtype(self._sim_params.dtype)

    # ------------------------------------------------------------------ #
    # Snapshot                                                           #
    # ------------------------------------------------------------------ #

    def snapshot(self) -> GbmTrainerConfig:
        return GbmTrainerConfig(
            cfg=self._mc_engine.snapshot(),
            domain_bounds=self._domain_bounds,
            cvnn=self._cvnn,
            optimizer_state=self._optimizer_state,
            global_step=self._global_step,
        )

    # ------------------------------------------------------------------ #
    # Internal helpers                                                   #
    # ------------------------------------------------------------------ #

    def _simulate_fft(self, contract: BlackScholes.Inputs) -> cp.ndarray:
        prices = self._mc_engine.price(inputs=contract).put_price
        price_mat = prices.reshape(self._batches_per_mc_run, self._network_size)
        return cp.mean(cp.fft.fft(price_mat, axis=1), axis=0)

    def _torch_step(
        self,
        real_in: torch.Tensor,
        imag_in: torch.Tensor,
        targets: torch.Tensor,
        optimizer: optim.Adam,
    ) -> Tuple[torch.Tensor, float]:
        pred_r, pred_i = self._cvnn(real_in, imag_in)
        loss = nn.functional.mse_loss(pred_r, targets.real) + nn.functional.mse_loss(
            pred_i, targets.imag
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        grad_norm = float(
            torch.nn.utils.clip_grad_norm_(self._cvnn.parameters(), float("inf"))
        )
        return loss, grad_norm

    # ------------------------------------------------------------------ #
    # Training loop                                                      #
    # ------------------------------------------------------------------ #

    def train(
        self,
        *,
        num_batches: int,
        batch_size: int,
        learning_rate: float,
        logger: Optional[StepLogger] = None,
    ) -> None:
        if self._device.type != "cuda":
            raise RuntimeError("Training requires a CUDA-capable GPU.")

        adam = optim.Adam(self._cvnn.parameters(), lr=learning_rate)
        if self._optimizer_state is not None:
            adam.load_state_dict(self._optimizer_state.to_torch(device=self._device))

        self._cvnn.train()

        for _ in range(num_batches):
            tic = time.perf_counter()

            # Sobol sampling ---------------------------------------- #
            sobol_inputs = self._sampler.sample(batch_size)

            # MC pricing + FFT (CuPy) ------------------------------ #
            assert self._cupy_stream is not None  # mypy guard
            with self._cupy_stream:
                fft_buf = cp.asarray(
                    [self._simulate_fft(contract) for contract in sobol_inputs]
                )
            self._cupy_stream.synchronize()

            # Forward/backward (PyTorch) --------------------------- #
            assert self._torch_stream is not None  # mypy guard
            with torch.cuda.stream(self._torch_stream):
                targets = (
                    torch.utils.dlpack.from_dlpack(fft_buf.toDlpack())
                    .to(self._torch_cdtype)
                    .detach()
                )
                real_in, imag_in = _split_inputs(
                    sobol_inputs, dtype=self._torch_rdtype, device=self._device
                )
                loss, grad_norm_val = self._torch_step(real_in, imag_in, targets, adam)
            self._torch_stream.synchronize()

            # Logging ---------------------------------------------- #
            batch_time = time.perf_counter() - tic
            if logger is not None:
                logger(
                    StepMetrics(
                        step=self._global_step,
                        batch_time=batch_time,
                        loss=loss.item(),
                        grad_norm=grad_norm_val,
                        lr=float(adam.param_groups[0]["lr"]),
                        optimizer=adam,
                        optimizer_state=AdamOptimizerState.from_torch(
                            adam.state_dict()
                        ),
                        model=self._cvnn,
                    )
                )
            self._global_step += 1

        self._optimizer_state = AdamOptimizerState.from_torch(adam.state_dict())

    # ------------------------------------------------------------------ #
    # Inference                                                          #
    # ------------------------------------------------------------------ #

    def predict_price(
        self,
        inputs: List[BlackScholes.Inputs],
    ) -> List[BlackScholes.HostPricingResults]:
        if not inputs:
            return []

        self._cvnn.eval()
        real_in, imag_in = _split_inputs(
            inputs, dtype=self._torch_rdtype, device=self._device
        )

        if self._torch_stream is None:
            pred_r, pred_i = self._cvnn(real_in, imag_in)
        else:
            with torch.cuda.stream(self._torch_stream):
                pred_r, pred_i = self._cvnn(real_in, imag_in)
            self._torch_stream.synchronize()

        spectrum = torch.complex(pred_r, pred_i)
        avg_ifft = torch.fft.ifft(spectrum, dim=1).mean(dim=1)

        results: List[BlackScholes.HostPricingResults] = []
        for value, contract in zip(avg_ifft, inputs):
            real_val = float(torch.real(value).item())
            imag_val = float(torch.imag(value).item())
            if abs(imag_val) > 1.0e-6:
                warnings.warn(
                    f"IFFT imaginary component {imag_val:.3e} exceeds tolerance.",
                    RuntimeWarning,
                )
            put_price = real_val

            discount = math.exp(-contract.r * contract.T)
            forward = contract.X0 * math.exp((contract.r - contract.d) * contract.T)

            intrinsic_put = discount * max(contract.K - forward, 0.0)
            intrinsic_call = discount * max(forward - contract.K, 0.0)
            call_price = put_price + forward - contract.K * discount

            results.append(
                BlackScholes.HostPricingResults(
                    put_price_intrinsic=intrinsic_put,
                    call_price_intrinsic=intrinsic_call,
                    underlying=forward,
                    put_convexity=put_price - intrinsic_put,
                    call_convexity=call_price - intrinsic_call,
                    put_price=put_price,
                    call_price=call_price,
                )
            )

        return results
