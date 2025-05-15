"""spectralmc.gbm_trainer
========================
Train a complex-valued neural network (``CVNN``) that learns the discrete
Fourier transform (DFT) of discounted Black-Scholes put-payoff distributions and
simultaneously log rich diagnostics to TensorBoard — all in a form that passes
``mypy --strict``.

-------------------------------------------------------------------------------
Why this file?
-------------------------------------------------------------------------------
The original trainer worked, but *mypy* raised "untyped call" errors because the
TensorBoard API lacks typing stubs.  We adopt **option B**: disable just the
``no-untyped-call`` error code for this file (leaving every other strict rule
active) and provide full annotations everywhere else.

-------------------------------------------------------------------------------
Module-level constants
-------------------------------------------------------------------------------
IMAG_TOL
    Numerical tolerance on the imaginary part of the IFFT output.
DEFAULT_LEARNING_RATE
    Default learning rate for the Adam optimiser.
TRAIN_LOG_INTERVAL
    Mini-batch interval between console status messages.
LOG_DIR_DEFAULT
    Default root directory for TensorBoard event files.
"""

# mypy: disable-error-code=no-untyped-call

from __future__ import annotations

import math
import time
import warnings
from typing import List, Tuple

import cupy as cp  # type: ignore[import-untyped]
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from spectralmc.cvnn import CVNN
from spectralmc.gbm import BlackScholes, SimulationParams
from spectralmc.sobol_sampler import BoundSpec, SobolSampler

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

IMAG_TOL: float = 1e-6
DEFAULT_LEARNING_RATE: float = 1e-3
TRAIN_LOG_INTERVAL: int = 10
LOG_DIR_DEFAULT: str = ".logs/gbm_trainer"

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def _get_torch_dtype(sim_precision: str) -> torch.dtype:
    """Return the torch dtype that matches a simulation precision string.

    Args:
        sim_precision: Either ``"float32"`` or ``"float64"``.

    Returns:
        ``torch.float32`` or ``torch.float64``.

    Raises:
        ValueError: If *sim_precision* is neither supported value.
    """
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
    """Convert Black-Scholes inputs to real/imag tensors for the CVNN.

    Args:
        inputs: Contract parameter sets.
        dtype: Desired floating-point precision.
        device: Destination device (``cpu`` / ``cuda``).

    Returns:
        Tuple ``(real, imag)`` where each tensor has shape ``(len(inputs), 6)``.
    """
    param_names: List[str] = list(BlackScholes.Inputs.model_fields.keys())
    rows: List[List[float]] = [
        [float(getattr(inp, name)) for name in param_names] for inp in inputs
    ]
    real = torch.tensor(rows, dtype=dtype, device=device)
    imag = torch.zeros_like(real)
    return real, imag


class _TBLogger:
    """Facade around :class:`torch.utils.tensorboard.SummaryWriter`."""

    def __init__(
        self, logdir: str, hist_every: int, flush_every: int
    ) -> None:  # noqa: D401
        self._writer: SummaryWriter = SummaryWriter(log_dir=logdir)
        self._hist_every: int = max(1, hist_every)
        self._flush_every: int = max(1, flush_every)

    # ------------------------------------------------------------------ #
    # Public API                                                         #
    # ------------------------------------------------------------------ #

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
        """Log scalars and (periodically) weight/gradient histograms."""
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
        """Flush remaining events and close the writer."""
        self._writer.flush()
        self._writer.close()


# ---------------------------------------------------------------------------
# Main trainer class
# ---------------------------------------------------------------------------


class GbmTrainer:
    """Train a :class:`~spectralmc.cvnn.CVNN` on Monte-Carlo Black-Scholes data.

    The class orchestrates three subsystems:

    * **Sobol sampling** over the six Black-Scholes input dimensions.
    * **CuPy Monte-Carlo engine** to produce discounted pay-offs.
    * **TensorBoard logger** for real-time diagnostics.
    """

    # ------------------------------------------------------------------ #
    # Construction                                                       #
    # ------------------------------------------------------------------ #

    def __init__(
        self,
        *,
        sim_params: SimulationParams,
        domain_bounds: dict[str, BoundSpec],
        skip_sobol: int,
        sobol_seed: int,
        cvnn: CVNN,
        device: torch.device | None = None,
        tb_logdir: str = LOG_DIR_DEFAULT,
        hist_every: int = 10,
        flush_every: int = 100,
    ) -> None:
        self.sim_params = sim_params
        self.cvnn = cvnn
        self.skip_sobol = skip_sobol
        self.sobol_seed = sobol_seed

        # Determine device.
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        # Sobol sampler initialised for Black-Scholes Inputs.
        self.sampler = SobolSampler(
            pydantic_class=BlackScholes.Inputs,
            dimensions=domain_bounds,
            skip=skip_sobol,
            seed=sobol_seed,
        )

        # GPU-powered Black-Scholes pricer.
        self.mc_engine = BlackScholes(sim_params)

        # Move CVNN to device & correct dtype.
        self.cvnn.to(device=self.device, dtype=_get_torch_dtype(sim_params.dtype))

        # TensorBoard logger.
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
        """Optimise *cvnn* parameters on Monte-Carlo data.

        Args:
            num_batches: Number of optimisation steps.
            batch_size: Sobol points per step.
            learning_rate: Base Adam learning rate.

        Raises:
            RuntimeError: If called on a non-CUDA machine (CuPy required).
        """
        if self.device.type != "cuda":  # CuPy cannot run on CPU in this context
            raise RuntimeError("Training must be executed on a CUDA device.")

        optimiser: torch.optim.Adam = torch.optim.Adam(
            self.cvnn.parameters(), lr=learning_rate
        )
        self.cvnn.train()

        cupy_cdtype = (
            cp.complex64 if self.sim_params.dtype == "float32" else cp.complex128
        )
        torch_cdtype = (
            torch.complex64 if self.sim_params.dtype == "float32" else torch.complex128
        )
        torch_rdtype = _get_torch_dtype(self.sim_params.dtype)

        global_step: int = 0
        for step in range(1, num_batches + 1):
            t0 = time.perf_counter()

            # ------------------------------------------------------------------
            # 1) Sobol sample parameter grid
            # ------------------------------------------------------------------
            sobol_points: List[BlackScholes.Inputs] = self.sampler.sample(batch_size)

            # ------------------------------------------------------------------
            # 2) Allocate target payoff FFT matrix on GPU (CuPy)
            # ------------------------------------------------------------------
            payoff_fft_cp = cp.zeros(
                (batch_size, self.sim_params.network_size), dtype=cupy_cdtype
            )

            # ------------------------------------------------------------------
            # 3) Monte-Carlo pricing contract-by-contract to keep memory low
            # ------------------------------------------------------------------
            for i, contract in enumerate(sobol_points):
                mc_result = self.mc_engine.price(inputs=contract)
                put_prices_cp: cp.ndarray = mc_result.put_price  # shape (paths,)

                put_mat = put_prices_cp.reshape(
                    (self.sim_params.batches_per_mc_run, self.sim_params.network_size)
                )
                payoff_fft_cp[i, :] = cp.mean(cp.fft.fft(put_mat, axis=1), axis=0)

            # ------------------------------------------------------------------
            # 4) Move targets to Torch via DLPack (zero-copy)
            # ------------------------------------------------------------------
            targets = torch.utils.dlpack.from_dlpack(payoff_fft_cp.toDlpack()).to(
                torch_cdtype
            )

            # ------------------------------------------------------------------
            # 5) Forward + loss + backward + optimiser step
            # ------------------------------------------------------------------
            real_in, imag_in = _inputs_to_real_imag(
                sobol_points, torch_rdtype, self.device
            )
            pred_r, pred_i = self.cvnn(real_in, imag_in)

            loss = nn.functional.mse_loss(
                pred_r, targets.real
            ) + nn.functional.mse_loss(pred_i, targets.imag)

            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            optimiser.step()

            # ------------------------------------------------------------------
            # 6) TensorBoard diagnostics
            # ------------------------------------------------------------------
            grad_norm: float = torch.nn.utils.clip_grad_norm_(
                self.cvnn.parameters(), max_norm=float("inf")
            ).item()
            batch_time: float = time.perf_counter() - t0
            lr: float = optimiser.param_groups[0]["lr"]

            self._tb.log_step(
                model=self.cvnn,
                step=global_step,
                loss=loss.item(),
                lr=lr,
                grad_norm=grad_norm,
                batch_time=batch_time,
            )

            global_step += 1

            if step % TRAIN_LOG_INTERVAL == 0 or step == num_batches:
                print(
                    f"[TRAIN] step={step}/{num_batches} \t"
                    f"loss={loss.item():.6f} \t"
                    f"time={batch_time*1000:6.1f} ms"
                )

        # Final flush.
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

        self.cvnn.eval()
        rdtype = _get_torch_dtype(self.sim_params.dtype)
        real_in, imag_in = _inputs_to_real_imag(inputs, rdtype, self.device)

        with torch.no_grad():
            pred_r, pred_i = self.cvnn(real_in, imag_in)
        spectrum = torch.complex(pred_r, pred_i)

        mean_ifft: torch.Tensor = torch.fft.ifft(spectrum, dim=1).mean(dim=1)

        results: List[BlackScholes.HostPricingResults] = []
        for complex_val, bs in zip(mean_ifft, inputs):
            real_part: float = float(torch.real(complex_val).item())
            imag_part: float = float(torch.imag(complex_val).item())
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
