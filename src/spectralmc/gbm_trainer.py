"""
gbm_trainer.py
==============
Trains a complex-valued neural network (CVNN) to learn the Fourier transform (DFT)
of a GPU-discounted put payoff distribution under Black-Scholes. This integrates:

  * SobolSampler for generating 6D input samples (X0, K, T, r, d, v).
  * BlackScholes for GPU-based payoff simulation (already discounted).
  * CVNN to map real inputs -> complex DFT outputs.

The training pipeline:
  1. Sample inputs from SobolSampler (List[BlackScholes.Inputs]).
  2. For each input, call black_scholes.price(...) to get `PricingResults`,
     focusing on `put_price`. This is an array of shape (paths,).
  3. Reshape put_price to (batches_per_mc_run, network_size), do row-wise FFT
     (axis=1) in CuPy, then average across rows => shape (network_size,).
  4. Store that average-FFT in a row of a big CuPy array (of shape
     (batch_size, network_size)), then do a single no-memcpy conversion to a
     PyTorch complex tensor.
  5. Split that complex PyTorch array into real/imag parts, compare to the
     CVNN output (MSE on real & imag).
  6. Update the CVNN with Adam.

Inference via `predict_price()`:
  1. Forward pass in batch for a list of inputs.
  2. Combine the real & imag outputs into a complex (batch_size, network_size).
  3. Perform ifft on dim=1, then average across each row => put price.
  4. Build HostPricingResults for each item with standard put/call relationships.

**Important**: Since BlackScholes uses CUDA via Numba, we assert that
`device.type == "cuda"` before training. Inference can be done on CPU though.

This module is intended for import; see the test file for usage examples.
"""

from __future__ import annotations

import math
from typing import List, Optional

import cupy as cp  # type: ignore[import]  # CuPy is untyped
import torch
import torch.nn as nn

from spectralmc.sobol_sampler import SobolSampler, BoundSpec
from spectralmc.gbm import BlackScholes, SimulationParams
from spectralmc.cvnn import CVNN


def _inputs_to_real_imag(
    input_list: List[BlackScholes.Inputs],
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert a list of BlackScholes.Inputs to real & imaginary PyTorch tensors.

    This uses the declared Pydantic fields in *Inputs* to preserve natural ordering
    (X0, K, T, r, d, v). The real tensor has shape (batch_size, 6). The imaginary
    tensor is all zeros, same shape.

    Args:
        input_list: A list of BlackScholes.Inputs, each specifying the 6 parameters.
        dtype: The PyTorch dtype to use (float32 or float64).
        device: The torch.device to place the tensors on.

    Returns:
        (real_in, imag_in) each of shape (batch_size, 6).
    """
    batch_size = len(input_list)
    # To preserve the declared field order, we rely on model_fields in the Inputs class
    # which is an OrderedDict as of Pydantic 2.x
    param_names = list(
        BlackScholes.Inputs.model_fields.keys()
    )  # e.g. ["X0","K","T","r","d","v"]

    arr = []
    for inp in input_list:
        row = []
        for name in param_names:
            val = float(getattr(inp, name))
            row.append(val)
        arr.append(row)

    real_in = torch.tensor(arr, dtype=dtype, device=device)
    imag_in = torch.zeros_like(real_in)
    return real_in, imag_in


def _get_torch_dtype(sp_dtype: str) -> torch.dtype:
    """Map SimulationParams dtype ("float32" or "float64") to torch.dtype."""
    if sp_dtype == "float32":
        return torch.float32
    elif sp_dtype == "float64":
        return torch.float64
    else:
        raise ValueError(f"Unsupported dtype in SimulationParams: {sp_dtype!r}")


class GbmTrainer:
    """Trainer that learns the FFT of put payoffs from a GPU-based BlackScholes simulation.

    Attributes:
        sim_params (SimulationParams): Specifies MC paths, timesteps, seeds, and floats precision.
        domain_bounds (dict[str, BoundSpec]): Bounds for X0,K,T,r,d,v used in SobolSampler.
        skip_sobol (int): Burn-in for the Sobol sequence.
        sobol_seed (int): Scramble seed for Sobol generation.
        cvnn (CVNN): The complex-valued neural network to be trained.
        device (torch.device): The device for training (must be "cuda" for MC).
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
        Args:
            sim_params: Configuration for the GPU-based Monte Carlo engine.
            domain_bounds: Parameter bounds for each field of BlackScholes.Inputs.
            skip_sobol: How many Sobol points to skip initially (burn-in).
            sobol_seed: Scrambling seed for the Sobol sequence.
            cvnn: A complex-valued neural net that maps 6 real inputs -> network_size complex outputs.
            device: Torch device. Must be "cuda" to run MC simulation. Inference can be on CPU if needed.
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

        # Create the GPU-based BlackScholes
        self.bsm_engine = BlackScholes(self.sim_params)

        # Move CVNN to device
        self.cvnn.to(self.device)

    def train(
        self,
        num_batches: int,
        batch_size: int,
        learning_rate: float = 1e-3,
    ) -> None:
        """Train the CVNN on discounted put payoffs for multiple Sobol-sampled points.

        Steps per batch:
          1. Sample `batch_size` Inputs from SobolSampler.
          2. For each one, call self.bsm_engine.price(...) to get a PricingResults that includes
             discounted put_price (shape = total_paths).
          3. Reshape put_price to (batches_per_mc_run, network_size).
          4. Row-wise FFT (axis=1), then mean across rows => length=network_size vector
             (call it payoff_mean_fft).
          5. Store payoff_mean_fft in a CuPy array row i. After the loop, convert the entire
             CuPy array to a torch.complex64 or complex128 (depending on sim_params.dtype),
             shape = (batch_size, network_size).
          6. Separate real/imag for the MSE vs the CVNN output. Then do an Adam update.

        Args:
            num_batches: Number of iterations (batches) to train over.
            batch_size: Number of Sobol-sampled points in each batch.
            learning_rate: Adam step size.

        Raises:
            AssertionError: If self.device.type != 'cuda' (the MC simulation needs GPU).
        """
        # We require a GPU device for the MC simulation
        assert self.device.type == "cuda", "Training requires a CUDA device."

        optimizer = torch.optim.Adam(self.cvnn.parameters(), lr=learning_rate)
        self.cvnn.train()

        # Decide if we want complex64 or complex128 based on sim_params
        cupy_complex_dtype = (
            cp.complex64 if self.sim_params.dtype == "float32" else cp.complex128
        )
        torch_complex_dtype = (
            torch.complex64 if self.sim_params.dtype == "float32" else torch.complex128
        )
        torch_real_dtype = _get_torch_dtype(self.sim_params.dtype)

        for step in range(1, num_batches + 1):
            # 1) Sample points
            sobol_points = self.sampler.sample(batch_size)

            # We will accumulate row i's average FFT in a CuPy array of shape (batch_size, network_size).
            payoff_fft_cp = cp.zeros(
                (batch_size, self.sim_params.network_size), dtype=cupy_complex_dtype
            )

            # 2) For each point: price(...) -> put_price -> shape (total_paths,)
            for i, bs_input in enumerate(sobol_points):
                pr = self.bsm_engine.price(inputs=bs_input)  # GPU-based PricingResults
                # pr.put_price is shape (total_paths,)
                put_arr = pr.put_price  # type: ignore  # cp.ndarray
                # Reshape => (batches_per_mc_run, network_size)
                # user wants shape (batches_per_mc_run, network_size)
                # total_paths = batches_per_mc_run * network_size
                put_mat = put_arr.reshape(
                    (self.sim_params.batches_per_mc_run, self.sim_params.network_size)
                )

                # 3) Row-wise FFT => axis=1 => shape (batches_per_mc_run, network_size)
                put_fft = cp.fft.fft(put_mat, axis=1)
                # 4) Mean across rows => shape (network_size,)
                payoff_mean_fft = cp.mean(put_fft, axis=0)  # axis=0 => average rows
                payoff_fft_cp[i, :] = payoff_mean_fft

            # 5) Convert payoff_fft_cp -> Torch complex on GPU (no mem copy) via DLPack
            dlpack_capsule = payoff_fft_cp.toDlpack()
            payoff_fft_torch = torch.utils.dlpack.from_dlpack(dlpack_capsule).to(
                torch_complex_dtype
            )

            # 6) Now payoff_fft_torch has shape (batch_size, network_size) complex
            # We'll separate real/imag for MSE
            target_real = payoff_fft_torch.real
            target_imag = payoff_fft_torch.imag

            # Build real_in, imag_in
            real_in, imag_in = _inputs_to_real_imag(
                input_list=sobol_points,
                dtype=torch_real_dtype,
                device=self.device,
            )

            # Forward pass
            pred_real, pred_imag = self.cvnn(real_in, imag_in)

            # Loss
            loss_r = nn.functional.mse_loss(pred_real, target_real)
            loss_i = nn.functional.mse_loss(pred_imag, target_imag)
            loss_val = loss_r + loss_i

            optimizer.zero_grad()
            loss_val.backward()  # type: ignore  # untyped call
            optimizer.step()

            if step % 10 == 0 or step == num_batches:
                print(f"[TRAIN] step={step}/{num_batches}, loss={loss_val.item():.6f}")

    def predict_price(
        self, inputs_list: List[BlackScholes.Inputs]
    ) -> List[BlackScholes.HostPricingResults]:
        """
        Predict the put price (undiscounted) for each item in inputs_list by:
          1. Forward pass of CVNN => shape (batch_size, network_size) real+imag.
          2. Convert to a single complex tensor, do ifft on dim=1 => shape (batch_size, network_size).
          3. Take mean across dim=1 => shape (batch_size,).
          4. The real part is the put price (assert imag part near zero).
          5. Fill out HostPricingResults by replicating the discounting logic from BlackScholes
             for each input item.

        This method can run on either CPU or GPU (CVNN forward pass).
        The final HostPricingResults is computed in Python floats.

        Args:
            inputs_list: A list of BlackScholes.Inputs to evaluate.

        Returns:
            A list of BlackScholes.HostPricingResults, length == len(inputs_list).
        """
        self.cvnn.eval()
        batch_size = len(inputs_list)
        if batch_size == 0:
            return []

        # Build real_in, imag_in
        torch_real_dtype = _get_torch_dtype(self.sim_params.dtype)
        real_in, imag_in = _inputs_to_real_imag(
            input_list=inputs_list, dtype=torch_real_dtype, device=self.device
        )

        # Forward pass => shape (batch_size, network_size)
        with torch.no_grad():
            pred_r, pred_i = self.cvnn(real_in, imag_in)

        # Combine into one complex array
        pred_complex = torch.complex(pred_r, pred_i)
        # ifft along dim=1 => shape (batch_size, network_size)
        iffted = torch.fft.ifft(pred_complex, dim=1)
        # mean across dimension=1 => shape (batch_size,) complex
        mean_vals = iffted.mean(dim=1)

        results: List[BlackScholes.HostPricingResults] = []
        for i, bs_input in enumerate(inputs_list):
            # The put price is real( mean_vals[i] ), and we check imaginary is ~0
            put_price_val_cplx = mean_vals[i]
            if abs(put_price_val_cplx.imag.item()) > 1e-6:
                raise ValueError(
                    f"Predict Price: ifft imaginary part is not near zero: {put_price_val_cplx.imag.item():.6g}"
                )
            put_price_val = float(put_price_val_cplx.real.item())

            # Now replicate discounting logic for a single set:
            X0f = float(bs_input.X0)
            Kf = float(bs_input.K)
            Tf = float(bs_input.T)
            rf = float(bs_input.r)
            df = float(bs_input.d)
            vf = float(bs_input.v)

            disc_factor = math.exp(-rf * Tf)
            forward = X0f * math.exp((rf - df) * Tf)
            put_price_intr = disc_factor * max(Kf - forward, 0.0)
            # We'll back out call price via parity: call = put + forward - K*df
            call_price_val = put_price_val + forward - (Kf * disc_factor)
            call_price_intr = disc_factor * max(forward - Kf, 0.0)

            # underlying = forward? The original get_host_price used underlying=mean final
            # but let's do as BlackScholes does: 'underlying' is the average final price
            # but we can't do that here. We'll just store forward as a good proxy.
            # (User said "the same way that happens in blackscholes.price"
            # but we don't have path info here. We'll store forward.)
            # We'll store 'underlying' as forward to be consistent with ignoring final paths.
            underlying_approx = forward

            # build HostPricingResults
            hpr = BlackScholes.HostPricingResults(
                put_price_intrinsic=put_price_intr,
                call_price_intrinsic=call_price_intr,
                underlying=underlying_approx,
                put_convexity=put_price_val - put_price_intr,
                call_convexity=call_price_val - call_price_intr,
                put_price=put_price_val,
                call_price=call_price_val,
            )
            results.append(hpr)
        return results
