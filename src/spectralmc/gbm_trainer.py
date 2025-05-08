"""
gbm_trainer.py
==============
Trains a complex-valued neural network (CVNN) to learn the Fourier transform (DFT)
of a European call payoff distribution under a GPU-based Black-Scholes simulation.

Core Steps:
    1) Sample 6D (X0,K,T,r,d,v) points with SobolSampler, validated by BlackScholes.Inputs.
    2) For each sample, run GPU MC to get final prices -> payoff -> column-wise DFT.
    3) Store these DFTs in a PyTorch complex tensor as training targets.
    4) Feed real (and zero-imag) inputs into a CVNN that outputs real+imag components
       (matching the DFT shape).
    5) Use MSE loss over real+imag parts to train the network.
    6) For inference, interpret the 0th DFT frequency (real part) as the sum of payoffs,
       returning sum / network_size.

Usage:
    python gbm_trainer.py

Requirements:
    * sobol_sampler.py
    * gbm.py
    * cvnn.py
    * mypy --strict compliance
"""

from __future__ import annotations

import math
from typing import List, Optional

import cupy as cp  # type: ignore[import]  # (CuPy is untyped)
import torch
import torch.nn as nn
import torch.optim

from spectralmc.sobol_sampler import BoundSpec, SobolSampler
from spectralmc.gbm import BlackScholes, SimulationParams
from spectralmc.cvnn import CVNN


class GbmTrainer:
    """
    A trainer that uses:
      - SobolSampler to generate 6D Black-Scholes Inputs.
      - BlackScholes GPU MC for payoff simulations.
      - A user-supplied CVNN to learn the payoff DFT.

    Attributes:
        sim_params: SimulationParams controlling the MC engine (timesteps, etc.).
        domain_bounds: Dict mapping parameter names ("X0", "K", "T", "r", "d", "v") to BoundSpec.
        skip_sobol: How many Sobol samples to skip initially (burn-in).
        sobol_seed: Scramble seed for Sobol sampler.
        cvnn: A complex-valued neural network from cvnn.py that we train.
        device: Torch device (e.g., 'cuda') for computation.
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
            sim_params: Configuration (timesteps, network_size, etc.) for the Monte Carlo engine.
            domain_bounds: 6D domain for SobolSampler, matching BlackScholes.Inputs fields.
            skip_sobol: Burn-in steps for Sobol sequence.
            sobol_seed: Seed for scrambling the Sobol generator.
            cvnn: Complex-valued neural network to be trained on DFT outputs.
            device: PyTorch device to store model and do computations. Defaults to CUDA if available.
        """
        self.sim_params = sim_params
        self.domain_bounds = domain_bounds
        self.skip_sobol = skip_sobol
        self.sobol_seed = sobol_seed
        self.cvnn = cvnn
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        # Build the Sobol sampler
        self.sampler = SobolSampler(
            pydantic_class=BlackScholes.Inputs,
            dimensions=self.domain_bounds,
            skip=self.skip_sobol,
            seed=self.sobol_seed,
        )

        # Build the GPU-based BlackScholes pricer
        self.bsm_engine = BlackScholes(self.sim_params)

        # Move CVNN to device
        self.cvnn.to(self.device)

    def train(
        self,
        num_batches: int,
        batch_size: int,
        learning_rate: float = 1e-3,
    ) -> None:
        """
        Perform training over multiple mini-batches of Sobol-sampled points.

        For each batch:
          - Sample `batch_size` points from the SobolSampler.
          - Run GPU-based MC with black_scholes._simulate(...) to get final payoffs.
          - Reshape to (network_size, batches_per_mc_run).
          - Column-wise CuPy FFT and average columns => shape (network_size,).
          - Convert to PyTorch complex => store in a (batch_size, network_size) target.
          - CVNN forward pass (6 real inputs => output of shape (batch_size, network_size) real+imag).
          - MSE loss on real + imag parts => .backward().
          - Adam step.

        Args:
            num_batches: How many iterations (batches).
            batch_size: How many sobol points per batch.
            learning_rate: Learning rate for the Adam optimizer.
        """
        optimizer = torch.optim.Adam(self.cvnn.parameters(), lr=learning_rate)
        self.cvnn.train()

        for step in range(1, num_batches + 1):
            # Sample sobol points
            sobol_points = self.sampler.sample(
                batch_size
            )  # list of BlackScholes.Inputs

            # Prepare payoff DFT storage: shape (batch_size, network_size), complex64
            payoff_dft = torch.zeros(
                (batch_size, self.sim_params.network_size),
                dtype=torch.complex64,
            )

            # For each point, run MC, do DFT in CuPy
            for i, bs_input in enumerate(sobol_points):
                sim_res = self.bsm_engine._simulate(bs_input)
                final_prices = sim_res.sims[-1, :]  # shape (total_paths,)

                # payoff = max(final_prices - K, 0)
                payoff_cp = cp.maximum(final_prices - float(bs_input.K), 0.0)

                # reshape to (network_size, batches_per_mc_run)
                payoff_matrix = payoff_cp.reshape(
                    (self.sim_params.network_size, self.sim_params.batches_per_mc_run)
                )
                # column-wise DFT along axis=0
                payoff_fft = cp.fft.fft(payoff_matrix, axis=0)
                # average across columns => shape (network_size,)
                payoff_mean = cp.mean(payoff_fft, axis=1)  # complex64
                # Move to torch
                payoff_mean_torch_real = torch.from_numpy(payoff_mean.real.get())
                payoff_mean_torch_imag = torch.from_numpy(payoff_mean.imag.get())

                payoff_complex = torch.view_as_complex(
                    torch.stack(
                        (payoff_mean_torch_real, payoff_mean_torch_imag), dim=-1
                    )
                )
                payoff_dft[i, :] = payoff_complex

            # Build real + imag inputs for CVNN => shape (batch_size, 6)
            input_real = torch.zeros(
                (batch_size, 6), dtype=torch.float32, device=self.device
            )
            input_imag = torch.zeros(
                (batch_size, 6), dtype=torch.float32, device=self.device
            )
            for i, bs_input in enumerate(sobol_points):
                input_real[i, 0] = float(bs_input.X0)
                input_real[i, 1] = float(bs_input.K)
                input_real[i, 2] = float(bs_input.T)
                input_real[i, 3] = float(bs_input.r)
                input_real[i, 4] = float(bs_input.d)
                input_real[i, 5] = float(bs_input.v)

            # Move payoff_dft to device as well
            payoff_dft = payoff_dft.to(self.device)

            # Separate into real + imag => shape (batch_size, network_size)
            target_real = payoff_dft.real
            target_imag = payoff_dft.imag

            # Forward pass
            pred_real, pred_imag = self.cvnn(input_real, input_imag)

            # MSE on real + MSE on imag
            loss_real = nn.functional.mse_loss(pred_real, target_real)
            loss_imag = nn.functional.mse_loss(pred_imag, target_imag)
            loss_val = loss_real + loss_imag

            optimizer.zero_grad()
            loss_val.backward()  # type: ignore[no-untyped-call]
            optimizer.step()

            if step % 10 == 0 or step == num_batches:
                print(f"Batch {step}/{num_batches}, loss: {loss_val.item():.6f}")

    def predict_mean_payoff(
        self, X0: float, K: float, T: float, r: float, d: float, v: float
    ) -> float:
        """
        Do a forward pass of CVNN at a single input. The network outputs a length=network_size
        complex vector. We interpret the 0th index (DC component) of that vector's real part
        as the sum of payoffs, so mean = real/ network_size.

        Args:
            X0, K, T, r, d, v: Floats for each parameter.

        Returns:
            The predicted mean payoff as a float.
        """
        self.cvnn.eval()
        real_in = torch.tensor(
            [[X0, K, T, r, d, v]], dtype=torch.float32, device=self.device
        )
        imag_in = torch.zeros_like(real_in)

        with torch.no_grad():
            pred_r, pred_i = self.cvnn(real_in, imag_in)
            # shape = (1, network_size)
            sum_payoffs: float = float(pred_r[0, 0].item())
            mean_payoff: float = sum_payoffs / float(self.sim_params.network_size)
        return mean_payoff


if __name__ == "__main__":
    # Minimal example run
    from spectralmc.cvnn import CVNN

    # 1) Build sim_params
    sp = SimulationParams(
        timesteps=32,
        network_size=8,
        batches_per_mc_run=8,
        threads_per_block=128,
        mc_seed=999,
        buffer_size=1,
        dtype="float32",
        simulate_log_return=True,
        normalize_forwards=False,
    )

    # 2) Domain bounds for Sobol
    domain_example = {
        "X0": BoundSpec(lower=50.0, upper=150.0),
        "K": BoundSpec(lower=50.0, upper=150.0),
        "T": BoundSpec(lower=0.1, upper=2.0),
        "r": BoundSpec(lower=0.0, upper=0.1),
        "d": BoundSpec(lower=0.0, upper=0.05),
        "v": BoundSpec(lower=0.1, upper=0.5),
    }

    # 3) Build a CVNN
    net = CVNN(
        input_features=6,
        output_features=8,  # must match network_size
        hidden_features=16,
        num_residual_blocks=1,
    )

    # 4) Instantiate GbmTrainer
    trainer = GbmTrainer(
        sim_params=sp,
        domain_bounds=domain_example,
        skip_sobol=0,
        sobol_seed=42,
        cvnn=net,
    )

    # 5) Train
    trainer.train(num_batches=20, batch_size=4, learning_rate=1e-3)

    # 6) Predict mean payoff
    payoff_pred = trainer.predict_mean_payoff(
        X0=100.0, K=100.0, T=1.0, r=0.02, d=0.01, v=0.2
    )
    print("Predicted mean payoff (undiscounted):", payoff_pred)
