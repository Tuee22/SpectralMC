"""
gbm_trainer.py
==============
Trains a complex-valued neural network (CVNN) to learn the Fourier transform (DFT)
of a GPU-discounted put payoff distribution under Black-Scholes. This integrates:

  * SobolSampler for generating 6D input samples (X0, K, T, r, d, v).
  * BlackScholes for GPU-based payoff simulation (already discounted).
  * CVNN to map real inputs -> complex DFT outputs.

Main Steps:
-----------
1) Sample (X0, K, T, r, d, v) from SobolSampler (quasi-random).
2) Use BlackScholes on GPU to get discounted put_price (shape = total_paths).
3) Reshape => (batches_per_mc_run, network_size), row-wise FFT => axis=1,
   then mean => shape (network_size,). Store in a CuPy array (batch_size rows).
4) Convert that array once to PyTorch complex, do MSE with the CVNN output.
5) For inference, do the reverse: forward pass -> ifft -> average => put price.

Important:
----------
- Must use CUDA for training due to numba.cuda in BlackScholes.
- Inference can be on CPU or GPU.
- We unify all dtypes (either float32 or float64) via SimulationParams.dtype.

"""

from __future__ import annotations

import math
from typing import List, Optional

import cupy as cp  # type: ignore[import-untyped]
import torch
import torch.nn as nn

from spectralmc.sobol_sampler import SobolSampler, BoundSpec
from spectralmc.gbm import BlackScholes, SimulationParams
from spectralmc.cvnn import CVNN


def _get_torch_dtype(sp_dtype: str) -> torch.dtype:
    """Map SimulationParams dtype ("float32" or "float64") to torch.dtype.

    Args:
        sp_dtype (str): "float32" or "float64"

    Returns:
        torch.dtype: torch.float32 or torch.float64

    Raises:
        ValueError: If sp_dtype is not one of the recognized strings.
    """
    if sp_dtype == "float32":
        return torch.float32
    elif sp_dtype == "float64":
        return torch.float64
    raise ValueError(f"Unsupported dtype in SimulationParams: {sp_dtype!r}")


def _inputs_to_real_imag(
    input_list: List[BlackScholes.Inputs],
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert a list of BlackScholes.Inputs into real + imaginary PyTorch Tensors.

    Uses the declared field order in BlackScholes.Inputs:
      X0, K, T, r, d, v
    so the shape is (batch_size, 6) for both tensors. The imaginary part is zero.

    Args:
        input_list: A list of Inputs to convert.
        dtype: The target float type (torch.float32 or torch.float64).
        device: Where to allocate the resulting tensors (usually 'cuda').

    Returns:
        (real_in, imag_in), each shape (batch_size, 6).
    """
    param_names = list(
        BlackScholes.Inputs.model_fields.keys()
    )  # ["X0","K","T","r","d","v"]
    arr = [[float(getattr(inp, name)) for name in param_names] for inp in input_list]
    real_in = torch.tensor(arr, dtype=dtype, device=device)
    imag_in = torch.zeros_like(real_in)
    return real_in, imag_in


class GbmTrainer:
    """Trainer that learns the FFT of put payoffs from a GPU-based BlackScholes simulation.

    Attributes:
        sim_params (SimulationParams): GPU MC config (paths, timesteps, dtype, etc.).
        domain_bounds (dict[str, BoundSpec]): Bounds for (X0,K,T,r,d,v) in SobolSampler.
        skip_sobol (int): Burn-in for Sobol sequence.
        sobol_seed (int): Seed for Sobol scrambling.
        cvnn (CVNN): Complex-valued net for DFT of payoffs.
        device (torch.device): Torch device. Must be 'cuda' for training.
    """

    def __init__(
        self,
        sim_params: SimulationParams,
        domain_bounds: dict[str, BoundSpec],
        skip_sobol: int,
        sobol_seed: int,
        cvnn: CVNN,
        device: Optional[torch.device] = None,
    ) -> None:
        """
        Initializes a GbmTrainer by constructing:
          1) A SobolSampler for 6D Inputs.
          2) A GPU-based BlackScholes engine with the specified sim_params.
          3) Moves the CVNN onto the correct device + ensures dtype consistency.

        Args:
            sim_params: Configuration (timesteps, network_size, etc.) for GPU-based Monte Carlo.
            domain_bounds: Parameter bounds for (X0,K,T,r,d,v).
            skip_sobol: Sobol burn-in steps.
            sobol_seed: Scramble seed for Sobol sequence.
            cvnn: The complex-valued net.
            device: Torch device, must be 'cuda' to train. If None, picks 'cuda' if available else 'cpu'
        """
        self.sim_params = sim_params
        self.domain_bounds = domain_bounds
        self.skip_sobol = skip_sobol
        self.sobol_seed = sobol_seed
        self.cvnn = cvnn

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        # Build the sampler
        self.sampler = SobolSampler(
            pydantic_class=BlackScholes.Inputs,
            dimensions=self.domain_bounds,
            skip=self.skip_sobol,
            seed=self.sobol_seed,
        )

        # Build GPU-based BlackScholes
        self.bsm_engine = BlackScholes(self.sim_params)

        # Convert CVNN to correct device & dtype
        torch_real_dtype = _get_torch_dtype(self.sim_params.dtype)
        self.cvnn.to(device=self.device, dtype=torch_real_dtype)

    def train(
        self,
        num_batches: int,
        batch_size: int,
        learning_rate: float = 1e-3,
    ) -> None:
        """Train the CVNN on discounted put payoffs from multiple Sobol-sampled points.

        Steps per batch:
          1) Sample `batch_size` from SobolSampler.
          2) For each, call black_scholes.price(...) -> put_price (cp.ndarray).
          3) Reshape => row-wise FFT => mean => shape (network_size,). Store row in CuPy array.
          4) Convert CuPy->Torch complex once, do MSE with CVNN output, Adam step.

        Args:
            num_batches (int): How many training iterations to run.
            batch_size (int): Number of parameter sets (sobol points) per iteration.
            learning_rate (float): LR for Adam.

        Raises:
            AssertionError: If self.device.type != 'cuda' (MC code requires GPU).
        """
        assert self.device.type == "cuda", "Training requires a CUDA device."
        optimizer = torch.optim.Adam(self.cvnn.parameters(), lr=learning_rate)
        self.cvnn.train()

        # Decide complex dtypes from sim_params
        cupy_complex_dtype = (
            cp.complex64 if self.sim_params.dtype == "float32" else cp.complex128
        )
        torch_complex_dtype = (
            torch.complex64 if self.sim_params.dtype == "float32" else torch.complex128
        )
        torch_real_dtype = _get_torch_dtype(self.sim_params.dtype)

        for step in range(1, num_batches + 1):
            # 1) Sample from sobol
            sobol_points = self.sampler.sample(batch_size)

            # CuPy array => shape (batch_size, network_size) complex
            payoff_fft_cp = cp.zeros(
                (batch_size, self.sim_params.network_size),
                dtype=cupy_complex_dtype,
            )

            for i, bs_input in enumerate(sobol_points):
                # MC -> put_price => shape (total_paths,)
                pr = self.bsm_engine.price(inputs=bs_input)
                put_price_cp = pr.put_price  # discounted put array

                # Reshape => (batches_per_mc_run, network_size)
                put_mat = put_price_cp.reshape(
                    (self.sim_params.batches_per_mc_run, self.sim_params.network_size)
                )
                # row-wise FFT => axis=1
                put_fft = cp.fft.fft(put_mat, axis=1)
                # mean => shape (network_size,)
                payoff_mean_fft = cp.mean(put_fft, axis=0)
                payoff_fft_cp[i, :] = payoff_mean_fft

            # Convert once CuPy->Torch
            dlpack_capsule = payoff_fft_cp.toDlpack()
            payoff_fft_torch = torch.utils.dlpack.from_dlpack(dlpack_capsule)
            payoff_fft_torch = payoff_fft_torch.to(torch_complex_dtype)

            target_real = payoff_fft_torch.real
            target_imag = payoff_fft_torch.imag

            # Build real_in, imag_in
            real_in, imag_in = _inputs_to_real_imag(
                input_list=sobol_points,
                dtype=torch_real_dtype,
                device=self.device,
            )

            # Forward pass => MSE => step
            pred_r, pred_i = self.cvnn(real_in, imag_in)
            loss_r = nn.functional.mse_loss(pred_r, target_real)
            loss_i = nn.functional.mse_loss(pred_i, target_imag)
            loss_val = loss_r + loss_i

            optimizer.zero_grad()
            loss_val.backward()  # type: ignore[no-untyped-call]
            optimizer.step()

            if step % 10 == 0 or step == num_batches:
                print(f"[TRAIN] step={step}/{num_batches}, loss={loss_val.item():.6f}")

    def predict_price(
        self,
        inputs_list: List[BlackScholes.Inputs],
    ) -> List[BlackScholes.HostPricingResults]:
        """Predict undiscounted put prices for each input by inverse FFT.

        1) Forward pass => shape (batch_size, network_size) real+imag in Torch.
        2) ifft => (batch_size, network_size).
        3) mean each row => real part => put price (assert imaginary is near 0).
        4) Reconstruct discount logic for HostPricingResults.

        Args:
            inputs_list: A list of param sets (X0,K,T,r,d,v).

        Returns:
            List[BlackScholes.HostPricingResults] with length == len(inputs_list).
        """
        self.cvnn.eval()
        if len(inputs_list) == 0:
            return []

        torch_real_dtype = _get_torch_dtype(self.sim_params.dtype)
        real_in, imag_in = _inputs_to_real_imag(
            input_list=inputs_list,
            dtype=torch_real_dtype,
            device=self.device,
        )

        with torch.no_grad():
            pred_r, pred_i = self.cvnn(real_in, imag_in)
        pred_complex = torch.complex(pred_r, pred_i)

        iffted = torch.fft.ifft(pred_complex, dim=1)
        mean_vals = iffted.mean(dim=1)

        def _build_hpr(
            ifft_val: torch.Tensor, inp: BlackScholes.Inputs
        ) -> BlackScholes.HostPricingResults:
            real_part = float(torch.real(ifft_val).item())
            imag_part = float(torch.imag(ifft_val).item())
            if abs(imag_part) > 1e-6:
                raise ValueError(f"ifft imaginary part not near zero: {imag_part:.6g}")

            put_price_val = real_part

            disc_factor = math.exp(-inp.r * inp.T)
            forward = inp.X0 * math.exp((inp.r - inp.d) * inp.T)
            put_price_intr = disc_factor * max(inp.K - forward, 0.0)
            call_price_val = put_price_val + forward - (inp.K * disc_factor)
            call_price_intr = disc_factor * max(forward - inp.K, 0.0)

            return BlackScholes.HostPricingResults(
                put_price_intrinsic=put_price_intr,
                call_price_intrinsic=call_price_intr,
                underlying=forward,
                put_convexity=put_price_val - put_price_intr,
                call_convexity=call_price_val - call_price_intr,
                put_price=put_price_val,
                call_price=float(call_price_val),
            )

        return [_build_hpr(val, inp) for val, inp in zip(mean_vals, inputs_list)]
