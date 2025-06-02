# src/spectralmc/gbm_trainer.py
# mypy: disable-error-code=no-untyped-call
"""
GPU trainer that learns the discrete Fourier transform of discounted
Black–Scholes pay-off distributions with a complex-valued neural network
(**CVNN**).

Design highlights
-----------------
* **CUDA-only** – training raises if no GPU is available.
* **Dedicated streams** – CuPy FFTs and PyTorch autograd kernels run on
  separate CUDA streams to keep the default stream free.
* **Pluggable logging** – any callable matching :pydata:`StepLogger` can consume
  per-step metrics.  A drop-in :class:`TensorBoardLogger` implementation is
  provided.
* **Snapshot/restore** – the entire mutable state, including the Adam
  optimiser state, lives in :class:`GbmTrainerConfig`.  Snapshots are fully
  deterministic and can be round-tripped into a fresh :class:`GbmTrainer`.
"""

from __future__ import annotations

import math
import time
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeAlias

import cupy as cp
import torch
import torch.nn as nn
from pydantic import BaseModel
from torch.utils.tensorboard import SummaryWriter

from spectralmc.cvnn import CVNN
from spectralmc.gbm import BlackScholes, BlackScholesConfig
from spectralmc.sobol_sampler import BoundSpec, SobolSampler

# --------------------------------------------------------------------------- #
# Typing aliases                                                              #
# --------------------------------------------------------------------------- #

OptimizerState: TypeAlias = Dict[str, Any]
StepLogger: TypeAlias = Callable[["StepMetrics"], None]

# --------------------------------------------------------------------------- #
# Pydantic models                                                             #
# --------------------------------------------------------------------------- #


class GbmTrainerConfig(BaseModel):
    """Serialisable snapshot of a :class:`GbmTrainer`.

    Attributes
    ----------
    cfg
        Full Black–Scholes configuration.
    domain_bounds
        Sobol-sampler bounds for each GBM input dimension.
    cvnn
        Complex-valued neural network to train.
    optimizer_state
        Optional Adam `state_dict`.  If present, training resumes from it by
        default.
    """

    cfg: BlackScholesConfig
    domain_bounds: Dict[str, BoundSpec]
    cvnn: CVNN
    optimizer_state: Optional[OptimizerState] = None
    global_step: int = 0

    class Config:
        arbitrary_types_allowed = True  # CVNN is a torch.nn.Module


class StepMetrics(BaseModel):
    """Per-iteration metrics forwarded to a :pydata:`StepLogger`.

    Attributes
    ----------
    step
        Zero-based optimisation step index.
    batch_time
        Wall-clock duration of the step in seconds.
    loss
        Mean squared error (real + imag parts).
    grad_norm
        L2 norm of all gradients **after** clipping.
    lr
        Learning-rate of the first Adam parameter group.
    optimizer_state
        Full Adam `state_dict` *after* the step.
    model
        The CVNN in its current state (after the step).
    """

    step: int
    batch_time: float
    loss: float
    grad_norm: float
    lr: float
    optimizer_state: OptimizerState
    model: CVNN

    class Config:
        arbitrary_types_allowed = True  # model contains tensors


# --------------------------------------------------------------------------- #
# TensorBoard implementation of `StepLogger`                                  #
# --------------------------------------------------------------------------- #


class TensorBoardLogger:
    """Stream :class:`StepMetrics` to TensorBoard.

    Parameters
    ----------
    logdir
        Root directory for event files.
    hist_every
        Histogram logging cadence (steps).
    flush_every
        Event-file flush cadence (steps).
    """

    def __init__(
        self,
        *,
        logdir: str = ".logs/gbm_trainer",
        hist_every: int = 10,
        flush_every: int = 100,
    ) -> None:
        self._writer: SummaryWriter = SummaryWriter(log_dir=logdir)
        self._hist_every: int = max(1, hist_every)
        self._flush_every: int = max(1, flush_every)

    def __call__(self, metrics: StepMetrics) -> None:  # noqa: D401
        """Write *metrics* to TensorBoard."""
        writer = self._writer
        step_id: int = metrics.step

        writer.add_scalar("Loss/train", metrics.loss, step_id)
        writer.add_scalar("LR", metrics.lr, step_id)
        writer.add_scalar("GradNorm", metrics.grad_norm, step_id)
        writer.add_scalar("BatchTime", metrics.batch_time, step_id)

        if step_id % self._hist_every == 0:
            for name, param in metrics.model.named_parameters():
                writer.add_histogram(name, param, step_id)
                if param.grad is not None:
                    writer.add_histogram(f"{name}.grad", param.grad, step_id)

        if step_id % self._flush_every == 0:
            writer.flush()

    def close(self) -> None:
        """Flush and close the underlying :class:`SummaryWriter`."""
        self._writer.close()


# --------------------------------------------------------------------------- #
# Internal helpers                                                            #
# --------------------------------------------------------------------------- #


def _torch_dtype(precision: str) -> torch.dtype:
    """Return a torch dtype matching *precision* (`float32` or `float64`)."""
    return torch.float32 if precision == "float32" else torch.float64


def _split_inputs(
    inputs: List[BlackScholes.Inputs],
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert Pydantic inputs into *real* and *imag* tensors."""
    fields: List[str] = list(BlackScholes.Inputs.model_fields.keys())
    rows: List[List[float]] = [
        [float(getattr(inp, f)) for f in fields] for inp in inputs
    ]
    real = torch.tensor(rows, dtype=dtype, device=device)
    imag = torch.zeros_like(real)
    return real, imag


# --------------------------------------------------------------------------- #
# Main trainer                                                                #
# --------------------------------------------------------------------------- #


class GbmTrainer:
    """Coordinates MC pricing, FFT and CVNN optimisation."""

    # -------------------------- Construction --------------------------- #

    def __init__(self, cfg: GbmTrainerConfig) -> None:
        """Create a trainer from a snapshot."""
        self._sim_params = cfg.cfg.sim_params
        self._cvnn: CVNN = cfg.cvnn
        self._domain_bounds: Dict[str, BoundSpec] = cfg.domain_bounds
        self._optimizer_state: Optional[OptimizerState] = cfg.optimizer_state
        self._global_step: int = cfg.global_step

        # Device & CUDA streams
        self._device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self._torch_stream: Optional[torch.cuda.Stream]
        self._cupy_stream: Optional[cp.cuda.Stream]
        if self._device.type == "cuda":
            self._torch_stream = torch.cuda.Stream(device=self._device, priority=0)
            self._cupy_stream = cp.cuda.Stream(non_blocking=False)
        else:
            self._torch_stream = None
            self._cupy_stream = None

        # Sobol sampler & Black-Scholes engine
        self._sampler: SobolSampler[BlackScholes.Inputs] = SobolSampler(
            pydantic_class=BlackScholes.Inputs,
            dimensions=self._domain_bounds,
            skip=self._sim_params.skip,
            seed=self._sim_params.mc_seed,
        )
        self._mc_engine: BlackScholes = BlackScholes(cfg.cfg)

        # Move model to correct device/dtype
        self._cvnn.to(self._device, _torch_dtype(self._sim_params.dtype))

    # --------------------------- Snapshot ------------------------------ #

    def snapshot(self) -> GbmTrainerConfig:
        """Return a deterministic snapshot of the current state."""
        return GbmTrainerConfig(
            cfg=self._mc_engine.snapshot(),
            domain_bounds=self._domain_bounds,
            cvnn=self._cvnn,
            optimizer_state=self._optimizer_state,
            global_step=self._global_step,
        )

    # --------------------------- Training ------------------------------ #

    def train(
        self,
        *,
        num_batches: int,
        batch_size: int,
        learning_rate: float,
        logger: Optional[StepLogger] = None,
    ) -> None:
        """Optimise the CVNN on Monte-Carlo-generated spectra.

        Parameters
        ----------
        num_batches
            Number of optimisation iterations to execute.
        batch_size
            Sobol sample size per iteration.
        learning_rate
            Adam base learning-rate.  Ignored when an optimiser state (with its
            own learning-rate) is restored.
        optimizer_state
            Explicit Adam `state_dict` to restore.  If ``None``, the snapshot’s
            state is reused (if any).
        logger
            Optional callable receiving :class:`StepMetrics` every iteration.
        """
        if self._device.type != "cuda":
            raise RuntimeError("Training requires a CUDA device.")
        if self._torch_stream is None or self._cupy_stream is None:  # mypy guard
            raise RuntimeError("CUDA streams not initialised.")

        # ---- Initialise Adam ---------------------------------------- #
        adam = torch.optim.Adam(self._cvnn.parameters(), lr=learning_rate)
        if self._optimizer_state is not None:
            adam.load_state_dict(self._optimizer_state)

        self._cvnn.train()

        # ---- Dtypes ------------------------------------------------- #
        cp_cdtype = (
            cp.complex64 if self._sim_params.dtype == "float32" else cp.complex128
        )
        torch_cdtype = (
            torch.complex64 if self._sim_params.dtype == "float32" else torch.complex128
        )
        torch_rdtype = _torch_dtype(self._sim_params.dtype)

        # ---- Main loop --------------------------------------------- #
        for _ in range(num_batches):
            tic: float = time.perf_counter()

            # ---------- Sobol sampling ---------------------------- #
            sobol_inputs = self._sampler.sample(batch_size)

            # ---------- Monte-Carlo + FFT (CuPy) ------------------ #
            with self._cupy_stream:
                fft_buf = cp.zeros(
                    (batch_size, self._sim_params.network_size), dtype=cp_cdtype
                )
                for idx, contract in enumerate(sobol_inputs):
                    prices = self._mc_engine.price(inputs=contract).put_price
                    price_mat = prices.reshape(
                        self._sim_params.batches_per_mc_run,
                        self._sim_params.network_size,
                    )
                    fft_buf[idx] = cp.mean(cp.fft.fft(price_mat, axis=1), axis=0)
            self._cupy_stream.synchronize()

            # ---------- Forward/backward (PyTorch) ---------------- #
            with torch.cuda.stream(self._torch_stream):
                targets = (
                    torch.utils.dlpack.from_dlpack(fft_buf.toDlpack())
                    .to(torch_cdtype)
                    .detach()
                )
                real_in, imag_in = _split_inputs(
                    sobol_inputs, dtype=torch_rdtype, device=self._device
                )
                pred_r, pred_i = self._cvnn(real_in, imag_in)

                loss_r = nn.functional.mse_loss(pred_r, targets.real)
                loss_i = nn.functional.mse_loss(pred_i, targets.imag)
                loss = loss_r + loss_i

                adam.zero_grad(set_to_none=True)
                loss.backward()
                adam.step()

                grad_norm_val = float(
                    torch.nn.utils.clip_grad_norm_(
                        self._cvnn.parameters(), float("inf")
                    )
                )
            self._torch_stream.synchronize()

            # ---------- Logging ------------------------------------ #
            duration = time.perf_counter() - tic
            if logger is not None:
                logger(
                    StepMetrics(
                        step=self._global_step,
                        batch_time=duration,
                        loss=loss.item(),
                        grad_norm=grad_norm_val,
                        lr=float(adam.param_groups[0]["lr"]),
                        optimizer_state=adam.state_dict(),
                        model=self._cvnn,
                    )
                )

            self._global_step += 1

        # Keep final optimiser state for future snapshots
        self._optimizer_state = adam.state_dict()

    # --------------------------- Inference --------------------------- #

    def predict_price(
        self,
        inputs: List[BlackScholes.Inputs],
    ) -> List[BlackScholes.HostPricingResults]:
        """Return discounted put & call prices for *inputs*."""
        if not inputs:
            return []

        self._cvnn.eval()
        rdtype = _torch_dtype(self._sim_params.dtype)
        real_in, imag_in = _split_inputs(inputs, dtype=rdtype, device=self._device)

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
