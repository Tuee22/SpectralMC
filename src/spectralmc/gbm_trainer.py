# spectralmc/gbm_trainer.py
# mypy: disable-error-code=no-untyped-call
"""
Train a complex-valued neural network (``CVNN``) to learn the discrete Fourier
transform of discounted Black-Scholes put-pay-off distributions while logging
diagnostics to TensorBoard – all under ``mypy --strict``.

Key implementation details
--------------------------
* **GPU-only training** – `train()` raises unless the selected device is CUDA.
* **Dedicated CUDA streams**
  * **CuPy stream**  → ``self._cupy_stream`` (Monte-Carlo pricing & FFT).
  * **Torch stream** → ``self._torch_stream`` (all PyTorch ops).
  Each trainer keeps its work off the device-default stream.
* **CPU/GPU inference** – `predict_price()` also works on CPU; on CPU builds
  neither stream is created and the code falls back to synchronous execution.
"""

from __future__ import annotations

import math
import time
import warnings
from typing import List, Tuple

import cupy as cp
import torch
import torch.nn as nn
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

# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #


def _get_torch_dtype(sim_precision: str) -> torch.dtype:
    """Return the torch dtype that matches a simulation precision string."""
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


class _TBLogger:
    """Thin façade around :class:`torch.utils.tensorboard.SummaryWriter`."""

    def __init__(self, logdir: str, hist_every: int, flush_every: int) -> None:
        self._writer: SummaryWriter = SummaryWriter(log_dir=logdir)
        self._hist_every: int = max(1, hist_every)
        self._flush_every: int = max(1, flush_every)

    def log_step(
        self,
        *,
        model: nn.Module,
        step: int,
        loss: float,
        lr: float,
        grad_norm: float,
        batch_time: float,
    ) -> None:
        w = self._writer
        w.add_scalar("Loss/train", loss, step)
        w.add_scalar("LR", lr, step)
        w.add_scalar("GradNorm", grad_norm, step)
        w.add_scalar("BatchTime", batch_time, step)

        if step % self._hist_every == 0:
            for name, param in model.named_parameters():
                w.add_histogram(name, param, step)
                if param.grad is not None:
                    w.add_histogram(f"{name}.grad", param.grad, step)

        if step % self._flush_every == 0:
            w.flush()

    def close(self) -> None:
        self._writer.flush()
        self._writer.close()


# --------------------------------------------------------------------------- #
# Main trainer                                                                #
# --------------------------------------------------------------------------- #


class GbmTrainer:
    """Train a :pyclass:`spectralmc.cvnn.CVNN` on GPU Monte-Carlo GBM data."""

    # ------------------------------------------------------------------ #
    # Construction                                                       #
    # ------------------------------------------------------------------ #

    def __init__(
        self,
        *,
        cfg: BlackScholesConfig,
        domain_bounds: dict[str, BoundSpec],
        cvnn: CVNN,
        tb_logdir: str = LOG_DIR_DEFAULT,
        hist_every: int = 10,
        flush_every: int = 100,
    ) -> None:
        # Persist configuration -----------------------------------------------
        self._cfg = cfg
        self._cvnn = cvnn

        # Device selection -----------------------------------------------------
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Per-trainer CUDA streams (only on GPU) -------------------------------
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

        # Sobol sampler --------------------------------------------------------
        self._sampler = SobolSampler(
            pydantic_class=BlackScholes.Inputs,
            dimensions=domain_bounds,
            skip=self._cfg.sim_params.skip,
            seed=self._cfg.sim_params.mc_seed,
        )

        # Black-Scholes MC engine ---------------------------------------------
        self._mc_engine = BlackScholes(cfg)

        # Move CVNN to device/dtype -------------------------------------------
        self._cvnn.to(
            device=self._device, dtype=_get_torch_dtype(self._cfg.sim_params.dtype)
        )

        # TensorBoard logger ---------------------------------------------------
        self._tb = _TBLogger(
            logdir=tb_logdir, hist_every=hist_every, flush_every=flush_every
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
    ) -> None:
        """Optimise *cvnn* parameters on Monte-Carlo data."""
        if self._device.type != "cuda":
            raise RuntimeError("Training must be executed on a CUDA device.")
        assert self._torch_stream is not None
        assert self._cupy_stream is not None

        optimiser = torch.optim.Adam(self._cvnn.parameters(), lr=learning_rate)
        self._cvnn.train()

        cupy_cdtype = (
            cp.complex64 if self._cfg.sim_params.dtype == "float32" else cp.complex128
        )
        torch_cdtype = (
            torch.complex64
            if self._cfg.sim_params.dtype == "float32"
            else torch.complex128
        )
        torch_rdtype = _get_torch_dtype(self._cfg.sim_params.dtype)

        global_step = 0
        for step in range(1, num_batches + 1):
            step_start = time.perf_counter()

            # 1) Sobol sample inputs ------------------------------------------
            sobol_points = self._sampler.sample(batch_size)

            # 2-3) Monte-Carlo pricing & FFT – in the CuPy stream -------------
            with self._cupy_stream:
                payoff_fft_cp = cp.zeros(
                    (batch_size, self._cfg.sim_params.network_size), dtype=cupy_cdtype
                )

                for i, contract in enumerate(sobol_points):
                    mc_result = self._mc_engine.price(inputs=contract)
                    put_prices_cp: cp.ndarray = mc_result.put_price

                    put_mat = put_prices_cp.reshape(
                        (
                            self._cfg.sim_params.batches_per_mc_run,
                            self._cfg.sim_params.network_size,
                        )
                    )
                    payoff_fft_cp[i] = cp.mean(cp.fft.fft(put_mat, axis=1), axis=0)

            # 4) Barrier – wait for CuPy work to finish ------------------------
            self._cupy_stream.synchronize()

            # 5) PyTorch work – in the Torch stream ---------------------------
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

            # 6) TensorBoard ---------------------------------------------------
            batch_time = time.perf_counter() - step_start
            lr = optimiser.param_groups[0]["lr"]

            self._tb.log_step(
                model=self._cvnn,
                step=global_step,
                loss=loss.item(),
                lr=lr,
                grad_norm=grad_norm,
                batch_time=batch_time,
            )
            global_step += 1

            if step % TRAIN_LOG_INTERVAL == 0 or step == num_batches:
                print(
                    f"[TRAIN] step={step}/{num_batches}  "
                    f"loss={loss.item():.6g}  "
                    f"time={batch_time*1000:6.1f} ms"
                )

        self._tb.close()

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
        rdtype = _get_torch_dtype(self._cfg.sim_params.dtype)
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
