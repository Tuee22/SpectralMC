# spectralmc/gbm_trainer.py
"""
gbm_trainer.py
==============
Trains a complex-valued neural network (CVNN) to learn the Fourier transform
(DFT) of discounted put-payoff distributions under Black-Scholes.

Main Steps
----------
1. Sobol-sample (X0, K, T, r, d, v).
2. GPU Monte-Carlo → discounted put prices.
3. Reshape → FFT (row-wise) → mean – gives one DFT vector per Sobol point.
4. Minimise MSE between CVNN output and Monte-Carlo target.
5. Inference: CVNN → IFFT → average → put price.

Constants
---------
IMAG_TOL              – tolerance for residual imag part after IFFT.
DEFAULT_LEARNING_RATE – default Adam learning rate.
TRAIN_LOG_INTERVAL    – print loss every *n* steps.
"""

from __future__ import annotations

import math
import warnings
from typing import List, Optional

import cupy as cp  # type: ignore[import-untyped]
import torch
import torch.nn as nn

from spectralmc.sobol_sampler import SobolSampler, BoundSpec
from spectralmc.gbm import BlackScholes, SimulationParams
from spectralmc.cvnn import CVNN

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

IMAG_TOL: float = 1e-6
DEFAULT_LEARNING_RATE: float = 1e-3
TRAIN_LOG_INTERVAL: int = 10

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def _get_torch_dtype(sp_dtype: str) -> torch.dtype:
    if sp_dtype == "float32":
        return torch.float32
    if sp_dtype == "float64":
        return torch.float64
    raise ValueError(f"Unsupported dtype in SimulationParams: {sp_dtype!r}")


def _inputs_to_real_imag(
    input_list: List[BlackScholes.Inputs],
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    param_names = list(BlackScholes.Inputs.model_fields.keys())  # X0, K, T, r, d, v
    arr = [[float(getattr(inp, name)) for name in param_names] for inp in input_list]
    real = torch.tensor(arr, dtype=dtype, device=device)
    imag = torch.zeros_like(real)
    return real, imag


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


class GbmTrainer:
    """End-to-end trainer that fits a CVNN to Monte-Carlo put-price spectra."""

    def __init__(
        self,
        sim_params: SimulationParams,
        domain_bounds: dict[str, BoundSpec],
        skip_sobol: int,
        sobol_seed: int,
        cvnn: CVNN,
        device: Optional[torch.device] = None,
    ) -> None:
        self.sim_params = sim_params
        self.cvnn = cvnn
        self.skip_sobol = skip_sobol
        self.sobol_seed = sobol_seed

        # pick device
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        # Sobol sampler
        self.sampler = SobolSampler(
            pydantic_class=BlackScholes.Inputs,
            dimensions=domain_bounds,
            skip=skip_sobol,
            seed=sobol_seed,
        )

        # GPU Monte-Carlo engine
        self.bsm_engine = BlackScholes(sim_params)

        # CVNN device/dtype
        self.cvnn.to(device=self.device, dtype=_get_torch_dtype(sim_params.dtype))

    # ------------------------------------------------------------------ #
    # Training                                                           #
    # ------------------------------------------------------------------ #

    def train(
        self,
        num_batches: int,
        batch_size: int,
        learning_rate: float = DEFAULT_LEARNING_RATE,
    ) -> None:
        """Mini-batch optimisation using Adam (double for-loop to keep GPU RAM low)."""
        assert self.device.type == "cuda", "Training requires a CUDA device."

        optimizer = torch.optim.Adam(self.cvnn.parameters(), lr=learning_rate)
        self.cvnn.train()

        cupy_complex_dtype = (
            cp.complex64 if self.sim_params.dtype == "float32" else cp.complex128
        )
        torch_complex_dtype = (
            torch.complex64 if self.sim_params.dtype == "float32" else torch.complex128
        )
        torch_real_dtype = _get_torch_dtype(self.sim_params.dtype)

        for step in range(1, num_batches + 1):
            # 1) Sobol sample
            sobol_points = self.sampler.sample(batch_size)

            # 2) Allocate target matrix on GPU
            payoff_fft_cp = cp.zeros(
                (batch_size, self.sim_params.network_size), dtype=cupy_complex_dtype
            )

            # 3) Monte-Carlo one parameter set at a time (keeps memory footprint low)
            for i, inp in enumerate(sobol_points):
                pr = self.bsm_engine.price(inputs=inp)
                put_prices = pr.put_price  # shape (total_paths,)

                put_mat = put_prices.reshape(
                    (
                        self.sim_params.batches_per_mc_run,
                        self.sim_params.network_size,
                    )
                )
                payoff_fft_cp[i, :] = cp.mean(cp.fft.fft(put_mat, axis=1), axis=0)

            # 4) CuPy → Torch (one conversion)
            targets = torch.utils.dlpack.from_dlpack(payoff_fft_cp.toDlpack()).to(
                torch_complex_dtype
            )

            # 5) Prepare inputs
            real_in, imag_in = _inputs_to_real_imag(
                sobol_points, torch_real_dtype, self.device
            )

            pred_r, pred_i = self.cvnn(real_in, imag_in)
            loss = nn.functional.mse_loss(
                pred_r, targets.real
            ) + nn.functional.mse_loss(pred_i, targets.imag)

            optimizer.zero_grad()
            loss.backward()  # type: ignore[arg-type]
            optimizer.step()

            if step % TRAIN_LOG_INTERVAL == 0 or step == num_batches:
                print(f"[TRAIN] step={step}/{num_batches}, loss={loss.item():.6f}")

    # ------------------------------------------------------------------ #
    # Inference                                                          #
    # ------------------------------------------------------------------ #

    def predict_price(
        self,
        inputs_list: List[BlackScholes.Inputs],
    ) -> List[BlackScholes.HostPricingResults]:
        """Return put prices on host for a list of parameter sets."""
        self.cvnn.eval()
        if not inputs_list:
            return []

        real_in, imag_in = _inputs_to_real_imag(
            inputs_list, _get_torch_dtype(self.sim_params.dtype), self.device
        )

        with torch.no_grad():
            pred_r, pred_i = self.cvnn(real_in, imag_in)
        preds = torch.complex(pred_r, pred_i)

        mean_vals = torch.fft.ifft(preds, dim=1).mean(dim=1)

        def _build_pricing_result(
            ifft_val: torch.Tensor,
            inp: BlackScholes.Inputs,
        ) -> BlackScholes.HostPricingResults:
            real_part = float(torch.real(ifft_val).item())
            imag_part = float(torch.imag(ifft_val).item())
            if abs(imag_part) > IMAG_TOL:
                warnings.warn(
                    f"IFFT imag part not near zero: {imag_part:.6g}",
                    RuntimeWarning,
                )

            disc = math.exp(-inp.r * inp.T)
            fwd = inp.X0 * math.exp((inp.r - inp.d) * inp.T)

            put_intr = disc * max(inp.K - fwd, 0.0)
            call_val = real_part + fwd - inp.K * disc
            call_intr = disc * max(fwd - inp.K, 0.0)

            return BlackScholes.HostPricingResults(
                put_price_intrinsic=put_intr,
                call_price_intrinsic=call_intr,
                underlying=fwd,
                put_convexity=real_part - put_intr,
                call_convexity=call_val - call_intr,
                put_price=real_part,
                call_price=call_val,
            )

        return [
            _build_pricing_result(val, inp) for val, inp in zip(mean_vals, inputs_list)
        ]
