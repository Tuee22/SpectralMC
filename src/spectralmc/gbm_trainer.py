# src/spectralmc/gbm_trainer.py
# mypy: disable-error-code=no-untyped-call
"""
GPU trainer for learning the Fourier spectrum of discounted Black-Scholes
pay-offs with a complex-valued neural network (CVNN).

Highlights
----------
* **CUDA-only execution** with dedicated PyTorch/CuPy streams.
* **Stateless logging** – any callable conforming to :pydata:`StepLogger` can
  receive per-step metrics.  A ready-made :class:`TensorBoardLogger` is
  supplied for convenience.
* **Snapshot/restore** – **all** mutable state (including the Adam optimiser
  state) lives in :class:`GbmTrainerConfig`.  A snapshot can be round-tripped
  into a fresh trainer and will reproduce the original state bit-for-bit.

The entire module passes *mypy `--strict`*.
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
# Type aliases                                                                #
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
        Complete Black-Scholes configuration.
    domain_bounds
        Sobol-sampler bounds for each model input dimension.
    cvnn
        Complex-valued neural network to train.
    optimizer_state
        Optional Adam `state_dict`.  If supplied, training resumes from it
        unless an explicit `optimizer_state` argument is passed to
        :meth:`GbmTrainer.train`.
    """

    cfg: BlackScholesConfig
    domain_bounds: Dict[str, BoundSpec]
    cvnn: CVNN
    optimizer_state: Optional[OptimizerState] = None

    class Config:
        arbitrary_types_allowed = True  # CVNN is a torch.nn.Module


class StepMetrics(BaseModel):
    """Per-iteration metrics delivered to a :pydata:`StepLogger`.

    Attributes
    ----------
    step
        Zero-based optimisation step index.
    batch_time
        Wall-clock duration of the step (*seconds*).
    loss
        Mean squared error.
    grad_norm
        Global L2 norm of all gradients **after** clipping.
    lr
        Learning-rate for the first Adam parameter group.
    optimizer_state
        Full Adam `state_dict` **after** the step.
    model
        The CVNN in its *current* state.
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
# Optional TensorBoard implementation                                         #
# --------------------------------------------------------------------------- #


class TensorBoardLogger:
    """TensorBoard implementation of :pydata:`StepLogger`.

    Parameters
    ----------
    logdir
        Root directory for event files.
    hist_every
        Histogram logging cadence (*steps*).
    flush_every
        Event-file flush cadence (*steps*).
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
        """Stream *metrics* to TensorBoard."""
        w = self._writer
        s = metrics.step

        w.add_scalar("Loss/train", metrics.loss, s)
        w.add_scalar("LR", metrics.lr, s)
        w.add_scalar("GradNorm", metrics.grad_norm, s)
        w.add_scalar("BatchTime", metrics.batch_time, s)

        if s % self._hist_every == 0:
            for name, param in metrics.model.named_parameters():
                w.add_histogram(name, param, s)
                if param.grad is not None:
                    w.add_histogram(f"{name}.grad", param.grad, s)

        if s % self._flush_every == 0:
            w.flush()

    def close(self) -> None:
        """Flush & close the underlying :class:`SummaryWriter`."""
        self._writer.close()


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #


def _torch_dtype(precision: str) -> torch.dtype:
    """Return the torch dtype corresponding to *precision*."""
    return torch.float32 if precision == "float32" else torch.float64


def _split_inputs(
    inputs: List[BlackScholes.Inputs],
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert Pydantic inputs into real & imaginary tensors."""
    field_names: List[str] = list(BlackScholes.Inputs.model_fields.keys())
    data_rows: List[List[float]] = [
        [float(getattr(inp, f)) for f in field_names] for inp in inputs
    ]
    real_part = torch.tensor(data_rows, dtype=dtype, device=device)
    imag_part = torch.zeros_like(real_part)
    return real_part, imag_part


# --------------------------------------------------------------------------- #
# Main trainer                                                                #
# --------------------------------------------------------------------------- #


class GbmTrainer:
    """Coordinates Monte-Carlo pricing, FFT and CVNN optimisation."""

    # ------------------------------------------------------------------ #
    # Construction                                                       #
    # ------------------------------------------------------------------ #

    def __init__(self, cfg: GbmTrainerConfig) -> None:
        """Instantiate a trainer from *cfg*."""
        # Persistent config fragments
        self._sim_params = cfg.cfg.sim_params
        self._cvnn: CVNN = cfg.cvnn
        self._domain_bounds: Dict[str, BoundSpec] = cfg.domain_bounds
        self._optimizer_state: Optional[OptimizerState] = cfg.optimizer_state

        # Device & streams
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

        # Sampler & MC engine
        self._sampler: SobolSampler[BlackScholes.Inputs] = SobolSampler(
            pydantic_class=BlackScholes.Inputs,
            dimensions=self._domain_bounds,
            skip=self._sim_params.skip,
            seed=self._sim_params.mc_seed,
        )
        self._mc_engine: BlackScholes = BlackScholes(cfg.cfg)

        # Move model to device/dtype
        self._cvnn.to(self._device, _torch_dtype(self._sim_params.dtype))

    # ------------------------------------------------------------------ #
    # Snapshot                                                           #
    # ------------------------------------------------------------------ #

    def snapshot(self) -> GbmTrainerConfig:
        """Return a fully-deterministic snapshot of current state."""
        return GbmTrainerConfig(
            cfg=self._mc_engine.snapshot(),
            domain_bounds=self._domain_bounds,
            cvnn=self._cvnn,
            optimizer_state=self._optimizer_state,
        )

    # ------------------------------------------------------------------ #
    # Training loop                                                      #
    # ------------------------------------------------------------------ #

    def train(
        self,
        *,
        num_batches: int,
        batch_size: int,
        learning_rate: float = 1.0e-3,
        optimizer_state: Optional[OptimizerState] = None,
        logger: Optional[StepLogger] = None,
    ) -> None:
        """Optimise the CVNN on Monte-Carlo-generated spectra.

        Parameters
        ----------
        num_batches
            Number of optimisation iterations.
        batch_size
            Sobol sample size per iteration.
        learning_rate
            Base Adam learning-rate (ignored if *optimizer_state* overrides it).
        optimizer_state
            Explicit Adam `state_dict`.  If ``None`` the snapshot's state is
            reused (if any).
        logger
            Optional callable receiving :class:`StepMetrics` every step.
        """
        if self._device.type != "cuda":
            raise RuntimeError("Training requires a CUDA device.")

        if self._torch_stream is None or self._cupy_stream is None:
            # mypy safeguard – those are never None on CUDA
            raise RuntimeError("CUDA streams not initialised.")

        # ------------- Adam initialisation --------------------------- #
        adam = torch.optim.Adam(self._cvnn.parameters(), lr=learning_rate)
        state_to_load: Optional[OptimizerState] = (
            optimizer_state if optimizer_state is not None else self._optimizer_state
        )
        if state_to_load is not None:
            adam.load_state_dict(state_to_load)

        self._cvnn.train()

        # ------------- Dtypes & constants --------------------------- #
        cp_cdtype = (
            cp.complex64 if self._sim_params.dtype == "float32" else cp.complex128
        )
        torch_cdtype = (
            torch.complex64 if self._sim_params.dtype == "float32" else torch.complex128
        )
        torch_rdtype = _torch_dtype(self._sim_params.dtype)

        # ------------- Main optimisation loop ----------------------- #
        for step_idx in range(num_batches):
            tic: float = time.perf_counter()

            # --------- Sobol sampling ----------------------------- #
            sobol_batch = self._sampler.sample(batch_size)

            # --------- Monte-Carlo + FFT (CuPy) ------------------- #
            with self._cupy_stream:
                fft_buf = cp.zeros(
                    (batch_size, self._sim_params.network_size), dtype=cp_cdtype
                )
                for row, contract in enumerate(sobol_batch):
                    prices = self._mc_engine.price(inputs=contract).put_price
                    price_mat = prices.reshape(
                        self._sim_params.batches_per_mc_run,
                        self._sim_params.network_size,
                    )
                    fft_buf[row] = cp.mean(cp.fft.fft(price_mat, axis=1), axis=0)
            self._cupy_stream.synchronize()

            # --------- Forward, backward, Adam step (Torch) -------- #
            with torch.cuda.stream(self._torch_stream):
                targets = torch.utils.dlpack.from_dlpack(fft_buf.toDlpack()).to(
                    torch_cdtype
                )
                real_in, imag_in = _split_inputs(
                    sobol_batch, dtype=torch_rdtype, device=self._device
                )
                pred_r, pred_i = self._cvnn(real_in, imag_in)

                loss_real = nn.functional.mse_loss(pred_r, targets.real)
                loss_imag = nn.functional.mse_loss(pred_i, targets.imag)
                loss = loss_real + loss_imag

                adam.zero_grad(set_to_none=True)
                loss.backward()
                adam.step()

                grad_norm = float(
                    torch.nn.utils.clip_grad_norm_(
                        self._cvnn.parameters(), float("inf")
                    )
                )

            self._torch_stream.synchronize()
            step_time: float = time.perf_counter() - tic

            # --------- Logging ------------------------------------ #
            if logger is not None:
                logger(
                    StepMetrics(
                        step=step_idx,
                        batch_time=step_time,
                        loss=loss.item(),
                        grad_norm=grad_norm,
                        lr=float(adam.param_groups[0]["lr"]),
                        optimizer_state=adam.state_dict(),
                        model=self._cvnn,
                    )
                )

            if (step_idx + 1) % 10 == 0 or (step_idx + 1) == num_batches:
                print(
                    f"[TRAIN] step={step_idx + 1:>4}/{num_batches} "
                    f"loss={loss.item():.6g} "
                    f"time={step_time*1_000:.1f} ms"
                )

        # Persist Adam state for future snapshots
        self._optimizer_state = adam.state_dict()

    # ------------------------------------------------------------------ #
    # Inference                                                          #
    # ------------------------------------------------------------------ #

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

        # Forward pass on the model stream
        if self._torch_stream is None:
            pred_r, pred_i = self._cvnn(real_in, imag_in)
        else:
            with torch.cuda.stream(self._torch_stream):
                pred_r, pred_i = self._cvnn(real_in, imag_in)
            self._torch_stream.synchronize()

        spectrum = torch.complex(pred_r, pred_i)
        mean_ifft = torch.fft.ifft(spectrum, dim=1).mean(dim=1)

        results: List[BlackScholes.HostPricingResults] = []
        for value, contract in zip(mean_ifft, inputs):
            real_val = float(torch.real(value).item())
            imag_val = float(torch.imag(value).item())
            if abs(imag_val) > 1.0e-6:
                warnings.warn(
                    f"IFFT imaginary component {imag_val:.3e} exceeds tolerance.",
                    RuntimeWarning,
                    stacklevel=1,
                )

            discount = math.exp(-contract.r * contract.T)
            forward = contract.X0 * math.exp((contract.r - contract.d) * contract.T)

            intrinsic_put = discount * max(contract.K - forward, 0.0)
            intrinsic_call = discount * max(forward - contract.K, 0.0)
            call_price = real_val + forward - contract.K * discount

            results.append(
                BlackScholes.HostPricingResults(
                    put_price_intrinsic=intrinsic_put,
                    call_price_intrinsic=intrinsic_call,
                    underlying=forward,
                    put_convexity=real_val - intrinsic_put,
                    call_convexity=call_price - intrinsic_call,
                    put_price=real_val,
                    call_price=call_price,
                )
            )

        return results
