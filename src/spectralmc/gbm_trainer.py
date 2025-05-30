# spectralmc/gbm_trainer.py
# mypy: disable-error-code=no-untyped-call
"""
Training utilities for learning the discrete Fourier transform of discounted
Black-Scholes pay-off distributions with a complex-valued neural network
(``CVNN``).

The class :class:`~spectralmc.gbm_trainer.GbmTrainer` performs GPU-only
Monte-Carlo pricing, feeds spectra to a CVNN and optimises the network with the
Adam algorithm.  All device transfers are executed on *dedicated* CUDA streams
to avoid blocking the default stream.

Logging, checkpointing, metrics aggregation and similar concerns are delegated
to an *external* callable (see :pydata:`StepLogger`).  One convenient
implementation, :class:`~spectralmc.gbm_trainer.TensorBoardLogger`, is provided
in this module but **never** instantiated by the trainer – callers create the
logger they need and pass it to :pymeth:`GbmTrainer.train`.
"""

from __future__ import annotations

import math
import time
import warnings
from typing import Any, Callable, Dict, List, Tuple, TypeAlias

import cupy as cp
import torch
import torch.nn as nn
from pydantic import BaseModel
from torch.utils.tensorboard import SummaryWriter

from spectralmc.cvnn import CVNN
from spectralmc.gbm import BlackScholes, BlackScholesConfig
from spectralmc.sobol_sampler import BoundSpec, SobolSampler

# --------------------------------------------------------------------------- #
# Constants                                                                   #
# --------------------------------------------------------------------------- #

IMAG_TOL: float = 1e-6
DEFAULT_LEARNING_RATE: float = 1e-3
TRAIN_LOG_INTERVAL: int = 10
LOG_DIR_DEFAULT: str = ".logs/gbm_trainer"
HIST_EVERY_DEFAULT: int = 10
FLUSH_EVERY_DEFAULT: int = 100

# --------------------------------------------------------------------------- #
# Public typing                                                               #
# --------------------------------------------------------------------------- #

# A serialisable Adam state dictionary.
OptimizerState: TypeAlias = Dict[str, Any]

# A callable that consumes training metrics.
StepLogger: TypeAlias = Callable[["StepMetrics"], None]  # noqa: F821 (forward ref)


class GbmTrainerConfig(BaseModel):
    """Configuration for :class:`~spectralmc.gbm_trainer.GbmTrainer`.

    Attributes
    ----------
    cfg
        Complete Black-Scholes model configuration.
    domain_bounds
        Sobol-sampler bounds for each Black-Scholes input dimension.
    cvnn
        Complex-valued neural network to be trained.
    """

    cfg: BlackScholesConfig
    domain_bounds: Dict[str, BoundSpec]
    cvnn: CVNN

    class Config:
        arbitrary_types_allowed = True  # cvnn is a custom class


class StepMetrics(BaseModel):
    """Per-step training metrics passed to :pydata:`StepLogger`."""

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
# Optional TensorBoard implementation of ``StepLogger``                       #
# --------------------------------------------------------------------------- #


class TensorBoardLogger:
    """A :pydata:`StepLogger` that streams metrics to TensorBoard.

    Parameters
    ----------
    logdir
        Directory in which event files are written.
    hist_every
        Histogram-logging cadence (steps).
    flush_every
        Event-file flush cadence (steps).
    """

    def __init__(
        self,
        *,
        logdir: str = LOG_DIR_DEFAULT,
        hist_every: int = HIST_EVERY_DEFAULT,
        flush_every: int = FLUSH_EVERY_DEFAULT,
    ) -> None:
        self._writer: SummaryWriter = SummaryWriter(log_dir=logdir)
        self._hist_every: int = max(1, hist_every)
        self._flush_every: int = max(1, flush_every)

    def __call__(self, metrics: StepMetrics) -> None:  # noqa: D401 (imperative)
        """Write *metrics* to TensorBoard."""
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

    def close(self) -> None:  # optional; caller can ignore
        """Flush and close the underlying :class:`~torch.utils.tensorboard.SummaryWriter`."""
        self._writer.flush()
        self._writer.close()


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #


def _get_torch_dtype(sim_precision: str) -> torch.dtype:
    """Map ``"float32" | "float64"`` to the corresponding :pymod:`torch` dtype."""
    if sim_precision == "float32":
        return torch.float32
    if sim_precision == "float64":
        return torch.float64
    raise ValueError(f"Unsupported dtype: {sim_precision!r}")


def _inputs_to_real_imag(
    inputs: List[BlackScholes.Inputs],
    dtype: torch.dtype,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert Black-Scholes inputs to real/imag tensors for a CVNN."""
    param_names: List[str] = list(BlackScholes.Inputs.model_fields.keys())
    rows: List[List[float]] = [
        [float(getattr(inp, name)) for name in param_names] for inp in inputs
    ]
    real = torch.tensor(rows, dtype=dtype, device=device)
    imag = torch.zeros_like(real)
    return real, imag


# --------------------------------------------------------------------------- #
# Main trainer                                                                #
# --------------------------------------------------------------------------- #


class GbmTrainer:
    """Trainer for Monte-Carlo-generated GBM spectra.

    Parameters
    ----------
    config
        Complete trainer configuration.
    """

    # ------------------------------------------------------------------ #
    # Construction                                                       #
    # ------------------------------------------------------------------ #

    def __init__(self, config: GbmTrainerConfig) -> None:
        # Persist configuration fragments -----------------------------------
        self._sim_params = config.cfg.sim_params
        self._cvnn = config.cvnn
        self._domain_bounds = config.domain_bounds

        # Device selection ---------------------------------------------------
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Dedicated CUDA streams (GPU only) ----------------------------------
        if self._device.type == "cuda":
            self._torch_stream: torch.cuda.Stream | None = torch.cuda.Stream(
                device=self._device, priority=0
            )
            self._cupy_stream: cp.cuda.Stream | None = cp.cuda.Stream(
                non_blocking=False
            )
        else:
            self._torch_stream = None
            self._cupy_stream = None

        # Sobol sampler ------------------------------------------------------
        self._sampler = SobolSampler(
            pydantic_class=BlackScholes.Inputs,
            dimensions=self._domain_bounds,
            skip=self._sim_params.skip,
            seed=self._sim_params.mc_seed,
        )

        # Black-Scholes MC engine -------------------------------------------
        self._mc_engine = BlackScholes(config.cfg)

        # Move network to device/dtype --------------------------------------
        self._cvnn.to(
            device=self._device, dtype=_get_torch_dtype(self._sim_params.dtype)
        )

    # ------------------------------------------------------------------ #
    # Public helpers                                                     #
    # ------------------------------------------------------------------ #

    def snapshot(self) -> GbmTrainerConfig:
        """Return an immutable snapshot of the current configuration."""
        return GbmTrainerConfig(
            cfg=self._mc_engine.snapshot(),
            domain_bounds=self._domain_bounds,
            cvnn=self._cvnn,
        )

    # ------------------------------------------------------------------ #
    # Training loop                                                      #
    # ------------------------------------------------------------------ #

    def train(
        self,
        *,
        num_batches: int,
        batch_size: int,
        learning_rate: float = DEFAULT_LEARNING_RATE,
        optimizer_state: OptimizerState | None = None,
        logger: StepLogger | None = None,
    ) -> None:
        """Optimise CVNN parameters on Monte-Carlo data.

        Parameters
        ----------
        num_batches
            Total number of optimisation steps.
        batch_size
            Size of each Sobol input batch.
        learning_rate
            Initial Adam learning rate.
        optimizer_state
            Optional Adam state to resume from.
        logger
            Optional callable receiving :class:`StepMetrics` every step.
        """
        if self._device.type != "cuda":
            raise RuntimeError("Training must be executed on a CUDA device.")
        assert self._torch_stream is not None
        assert self._cupy_stream is not None

        optimiser = torch.optim.Adam(self._cvnn.parameters(), lr=learning_rate)
        if optimizer_state is not None:
            optimiser.load_state_dict(optimizer_state)

        self._cvnn.train()

        cupy_cdtype = (
            cp.complex64 if self._sim_params.dtype == "float32" else cp.complex128
        )
        torch_cdtype = (
            torch.complex64 if self._sim_params.dtype == "float32" else torch.complex128
        )
        torch_rdtype = _get_torch_dtype(self._sim_params.dtype)

        global_step = 0

        for step in range(1, num_batches + 1):
            step_start = time.perf_counter()

            # 1) Sobol sample inputs ----------------------------------------
            sobol_points = self._sampler.sample(batch_size)

            # 2-3) Monte-Carlo pricing & FFT – in the CuPy stream ----------
            with self._cupy_stream:
                payoff_fft_cp = cp.zeros(
                    (batch_size, self._sim_params.network_size), dtype=cupy_cdtype
                )

                for i, contract in enumerate(sobol_points):
                    mc_result = self._mc_engine.price(inputs=contract)
                    put_prices_cp = mc_result.put_price

                    put_mat = put_prices_cp.reshape(
                        (
                            self._sim_params.batches_per_mc_run,
                            self._sim_params.network_size,
                        )
                    )
                    payoff_fft_cp[i] = cp.mean(cp.fft.fft(put_mat, axis=1), axis=0)

            # 4) Barrier – wait for CuPy work to finish ----------------------
            self._cupy_stream.synchronize()

            # 5) PyTorch work – in the Torch stream -------------------------
            with torch.cuda.stream(self._torch_stream):
                targets = torch.utils.dlpack.from_dlpack(payoff_fft_cp.toDlpack()).to(
                    torch_cdtype
                )

                real_in, imag_in = _inputs_to_real_imag(
                    sobol_points, torch_rdtype, self._device
                )
                pred_r, pred_i = self._cvnn(real_in, imag_in)

                loss = nn.functional.mse_loss(
                    pred_r, targets.real, reduction="mean"
                ) + nn.functional.mse_loss(pred_i, targets.imag, reduction="mean")

                optimiser.zero_grad(set_to_none=True)
                loss.backward()
                optimiser.step()

                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self._cvnn.parameters(), max_norm=float("inf")
                ).item()

            # Optional: ensure timing/logging sees finished Torch work
            self._torch_stream.synchronize()

            # 6) Logging -----------------------------------------------------
            batch_time = time.perf_counter() - step_start
            lr = float(optimiser.param_groups[0]["lr"])

            if logger is not None:
                logger(
                    StepMetrics(
                        step=global_step,
                        batch_time=batch_time,
                        loss=loss.item(),
                        grad_norm=grad_norm,
                        lr=lr,
                        optimizer_state=optimiser.state_dict(),
                        model=self._cvnn,
                    )
                )
            global_step += 1

            if step % TRAIN_LOG_INTERVAL == 0 or step == num_batches:
                print(
                    f"[TRAIN] step={step}/{num_batches}  "
                    f"loss={loss.item():.6g}  "
                    f"time={batch_time*1000:6.1f} ms"
                )

    # ------------------------------------------------------------------ #
    # Inference                                                          #
    # ------------------------------------------------------------------ #

    def predict_price(
        self,
        inputs: List[BlackScholes.Inputs],
    ) -> List[BlackScholes.HostPricingResults]:
        """Infer discounted put & call prices for a batch of contracts."""
        if not inputs:
            return []

        self._cvnn.eval()
        rdtype = _get_torch_dtype(self._sim_params.dtype)
        real_in, imag_in = _inputs_to_real_imag(inputs, rdtype, self._device)

        with torch.no_grad():
            if self._torch_stream is None:
                pred_r, pred_i = self._cvnn(real_in, imag_in)
            else:
                with torch.cuda.stream(self._torch_stream):
                    pred_r, pred_i = self._cvnn(real_in, imag_in)
                self._torch_stream.synchronize()

        spectrum = torch.complex(pred_r, pred_i)
        mean_ifft = torch.fft.ifft(spectrum, dim=1).mean(dim=1)

        results: List[BlackScholes.HostPricingResults] = []
        for complex_val, bs in zip(mean_ifft, inputs):
            real_part = float(torch.real(complex_val).item())
            imag_part = float(torch.imag(complex_val).item())
            if abs(imag_part) > IMAG_TOL:
                warnings.warn(
                    f"IFFT imaginary component {imag_part:.3e} exceeds tolerance",
                    RuntimeWarning,
                )

            disc = math.exp(-bs.r * bs.T)
            fwd = bs.X0 * math.exp((bs.r - bs.d) * bs.T)

            put_intr = disc * max(bs.K - fwd, 0.0)
            call_val = real_part + fwd - bs.K * disc
            call_intr = disc * max(fwd - bs.K, 0.0)

            results.append(
                BlackScholes.HostPricingResults(
                    put_price_intrinsic=put_intr,
                    call_price_intrinsic=call_intr,
                    underlying=fwd,
                    put_convexity=real_part - put_intr,
                    call_convexity=call_val - call_intr,
                    put_price=real_part,
                    call_price=call_val,
                )
            )
        return results
