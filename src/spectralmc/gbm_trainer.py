"""
gbm_trainer.py
==============
Complex-Valued Neural Network for GBM Payoff DFT
------------------------------------------------
This module trains a complex-valued neural network (CVNN) to learn the
Fourier transform (DFT) of a European call payoff distribution under
a Geometric Brownian Motion (GBM) model. It uses Sobol quasi-random sampling
to cover the 6D Black-Scholes parameter space: (X0, K, T, r, d, v).

Key Steps:
    1) Draw Sobol points in [0,1]^6 and map to actual domain bounds.
    2) Simulate Monte Carlo payoffs (for each point) in one step:
       S_T = S0 * exp((r - d - 0.5 * v^2) * T + v * sqrt(T) * Z).
    3) Reshape payoffs into (network_size, batch) and take column-wise FFT
       to approximate the characteristic function / DFT of the payoff.
    4) Train a CVNN that maps the 6 normalized inputs to the complex DFT.
    5) Use MSE loss on real+imag parts of the DFT output.

Runtime GPU Architecture Note:
------------------------------
If your CUDA driver/PyTorch build does not support the installed GPU,
this script falls back to CPU to avoid runtime errors. See PyTorch docs
for how to install a build that supports your GPU architecture.

Usage:
    python -m spectralmc.gbm_trainer

Example:
    # If run directly, it trains for 200 steps (batch_size=16) on a small domain
    # Then prints a predicted (undiscounted) payoff for (X0=100, K=100, T=1, r=0.02, d=0.01, v=0.2)
"""

import math
from typing import Optional, Union, Sequence, Tuple, Dict

import torch
import torch.nn as nn
from torch import Tensor


def _select_device_fallback() -> torch.device:
    """Select a suitable device (CUDA if supported, else CPU).

    Checks whether CUDA is available and the major capability is >= 5.
    If not suitable, falls back to CPU.

    Returns:
        A torch.device object pointing to 'cuda' or 'cpu'.
    """
    if torch.cuda.is_available():
        major, _minor = torch.cuda.get_device_capability()
        # Arbitrary check: require a major capability >= 5
        if major >= 5:
            return torch.device("cuda")
    return torch.device("cpu")


class ComplexLinear(nn.Module):
    """A linear (fully-connected) layer for complex inputs and outputs.

    This layer maintains real and imaginary weight matrices (and optional biases)
    for an affine transformation on complex inputs.  Given an input z ∈ ℂ^d,
    the transformation is:
      output = (W_r + i·W_i) · z + (b_r + i·b_i).

    Args:
        in_features: Dimensionality of the complex input.
        out_features: Dimensionality of the complex output.
        bias: Whether to include complex bias (True by default).
        device: Optional device for parameter storage.
        dtype: Optional data type for parameter storage.
    """

    # We allow None as bias => Optional
    bias_real: Optional[nn.Parameter]
    bias_imag: Optional[nn.Parameter]

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device: Optional[Union[torch.device, str]] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        # Real/imag weight: shape (out_features, in_features)
        self.weight_real = nn.Parameter(
            torch.empty((out_features, in_features), device=device, dtype=dtype)
        )
        self.weight_imag = nn.Parameter(
            torch.empty((out_features, in_features), device=device, dtype=dtype)
        )
        nn.init.xavier_uniform_(self.weight_real)
        nn.init.xavier_uniform_(self.weight_imag)

        if bias:
            self.bias_real = nn.Parameter(
                torch.zeros((out_features,), device=device, dtype=dtype)
            )
            self.bias_imag = nn.Parameter(
                torch.zeros((out_features,), device=device, dtype=dtype)
            )
        else:
            self.bias_real = None
            self.bias_imag = None

    def forward(self, input: Tensor) -> Tensor:
        """Apply the complex linear transformation.

        Args:
            input: A (batch_size, in_features) real or complex tensor.

        Returns:
            Complex tensor of shape (batch_size, out_features).
        """
        if not torch.is_complex(input):
            # Convert real -> complex with 0 imaginary
            imaginary_part = torch.zeros_like(input)
            input = torch.complex(input, imaginary_part)

        # Split into real/imag
        x_real = input.real
        x_imag = input.imag

        # Real part of output = W_r x_real - W_i x_imag
        # Imag part of output = W_i x_real + W_r x_imag
        out_real = x_real @ self.weight_real.T - x_imag @ self.weight_imag.T
        out_imag = x_real @ self.weight_imag.T + x_imag @ self.weight_real.T

        if self.bias_real is not None and self.bias_imag is not None:
            out_real = out_real + self.bias_real
            out_imag = out_imag + self.bias_imag

        comp_out: Tensor = torch.complex(out_real, out_imag)
        return comp_out


class ComplexResidualBlock(nn.Module):
    """A simple residual block with two ComplexLinear layers and ReLU.

    Each block computes:
      z_out = z_in + ComplexLinear(ReLU(ComplexLinear(z_in))).

    The input and output have the same shape (batch_size, features).

    Args:
        features: Number of complex features in input/output.
        device: Device for layers.
        dtype: Data type for layers.
    """

    def __init__(
        self,
        features: int,
        device: Optional[Union[torch.device, str]] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        self.linear1 = ComplexLinear(features, features, device=device, dtype=dtype)
        self.linear2 = ComplexLinear(features, features, device=device, dtype=dtype)

    def forward(self, z_in: Tensor) -> Tensor:
        """Compute forward pass of the residual block.

        Args:
            z_in: Complex input of shape (batch_size, features).

        Returns:
            A complex tensor with the same shape, after the residual operation.
        """
        z1: Tensor = self.linear1(z_in)
        # ReLU on real and imaginary parts
        z1 = torch.complex(torch.relu(z1.real), torch.relu(z1.imag))
        z2: Tensor = self.linear2(z1)
        # Residual connection
        z_out: Tensor = z_in + z2
        return z_out


class OptionPricingCVNN(nn.Module):
    """A configurable complex network for DFT of payoffs.

    The network maps 6 real inputs (X0, K, T, r, d, v) to a complex vector
    of length `output_features`, using a hidden dimension and optional residual blocks.

    Args:
        output_features: Number of complex output features (e.g. network_size for DFT).
        hidden_features: Number of complex features in the hidden layers.
        num_res_layers: Number of residual blocks to include.
        device: Device for layer parameters.
        dtype: Data type for layer parameters.
    """

    def __init__(
        self,
        output_features: int,
        hidden_features: int,
        num_res_layers: int,
        device: Optional[Union[torch.device, str]] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        self.num_res_layers = num_res_layers

        # Input linear: 6 -> hidden
        self.input_linear = ComplexLinear(
            6, hidden_features, device=device, dtype=dtype
        )

        # Residual blocks
        self.res_blocks = nn.ModuleList(
            [
                ComplexResidualBlock(hidden_features, device=device, dtype=dtype)
                for _ in range(num_res_layers)
            ]
        )

        # Output linear: hidden -> output_features
        self.output_linear = ComplexLinear(
            hidden_features, output_features, device=device, dtype=dtype
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass from real(ish) input to complex DFT.

        Args:
            x: Float or complex tensor of shape (batch_size, 6).

        Returns:
            Complex tensor (batch_size, output_features).
        """
        z_in: Tensor
        if not torch.is_complex(x):
            z_in = torch.complex(x, torch.zeros_like(x))
        else:
            z_in = x

        # Input linear + ReLU
        z_hidden_0: Tensor = self.input_linear(z_in)
        z_hidden_0 = torch.complex(
            torch.relu(z_hidden_0.real), torch.relu(z_hidden_0.imag)
        )

        # Residual blocks
        z_res: Tensor = z_hidden_0
        for block in self.res_blocks:
            z_res = block(z_res)
            z_res = torch.complex(torch.relu(z_res.real), torch.relu(z_res.imag))

        # Output
        z_out_pre: Tensor = self.output_linear(z_res)
        z_out: Tensor = z_out_pre  # no final activation
        return z_out


class OptionPricerTrainer:
    """Trainer for learning the spectral representation (DFT) of call payoffs under GBM.

    Steps:
      1) Draw Sobol points in [0,1]^6.
      2) Map them to actual domain for (X0, K, T, r, d, v).
      3) Simulate MC payoffs for each param set and reshape to (network_size, batch).
      4) Compute column-wise FFT -> shape (network_size, batch).
      5) Transpose to (batch, network_size) for training with a CVNN via MSE on complex outputs.

    Args:
        domain_bounds: Either a dict with keys ["X0","K","T","r","d","v"]
          or a 6-tuple list specifying (lower, upper) in order (X0, K, T, r, d, v).
        network_size: The number of simulated paths (also the DFT size).
        hidden_features: The hidden dimension (number of complex features).
        num_res_layers: Number of complex residual blocks.
        scramble: Whether to scramble the Sobol sequence (True by default).
        sobol_seed: Seed for the Sobol engine.
        device: Device for training (defaults to auto).
        dtype: Torch data type for model.
    """

    def __init__(
        self,
        domain_bounds: Union[
            Dict[str, Tuple[float, float]], Sequence[Tuple[float, float]]
        ],
        network_size: int,
        hidden_features: int,
        num_res_layers: int,
        scramble: bool = True,
        sobol_seed: Optional[int] = None,
        device: Optional[Union[torch.device, str]] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        # Parse domain bounds into shape (6,)
        if isinstance(domain_bounds, dict):
            param_order = ["X0", "K", "T", "r", "d", "v"]
            lowers = [domain_bounds[k][0] for k in param_order]
            uppers = [domain_bounds[k][1] for k in param_order]
        else:
            if len(domain_bounds) != 6:
                raise ValueError("Expected 6 parameter bounds for (X0, K, T, r, d, v).")
            lowers = [db[0] for db in domain_bounds]
            uppers = [db[1] for db in domain_bounds]

        if device is None:
            device = _select_device_fallback()

        self.device = device
        self.domain_lower = torch.tensor(
            lowers, dtype=torch.float32, device=self.device
        )
        self.domain_upper = torch.tensor(
            uppers, dtype=torch.float32, device=self.device
        )

        self.network_size = network_size
        self.model = OptionPricingCVNN(
            output_features=network_size,
            hidden_features=hidden_features,
            num_res_layers=num_res_layers,
            device=self.device,
            dtype=dtype,
        ).to(self.device)

        # We'll add a mypy ignore for the untyped SobolEngine
        self.sobol_engine = torch.quasirandom.SobolEngine(
            dimension=6, scramble=scramble, seed=sobol_seed
        )  # type: ignore[no-untyped-call]

    def _simulate_payoffs(self, params: Tensor) -> Tensor:
        """Simulate call payoffs for given parameter sets, shape (network_size, batch).

        A single-step GBM approach:
          S_T = S0 * exp((r - d - 0.5 * v^2)*T + v * sqrt(T) * Z)
          payoff = max(S_T - K, 0).

        Args:
            params: shape (batch, 6) -> [X0, K, T, r, d, v].

        Returns:
            A float Tensor of shape (network_size, batch).
        """
        batch_size = params.shape[0]
        S0 = params[:, 0]
        K = params[:, 1]
        T = params[:, 2]
        r = params[:, 3]
        div = params[:, 4]
        vol = params[:, 5]

        Z = torch.randn(
            (self.network_size, batch_size),
            device=self.device,
            dtype=torch.float32,
        )

        drift = (r - div - 0.5 * vol * vol) * T
        diffusion_scale = vol * torch.sqrt(torch.clamp(T, min=1e-12))

        drift_2d = drift.unsqueeze(0)
        diff_2d = diffusion_scale.unsqueeze(0)
        S0_2d = S0.unsqueeze(0)
        K_2d = K.unsqueeze(0)

        exponents = drift_2d + diff_2d * Z
        S_T = S0_2d * torch.exp(exponents)
        payoffs: Tensor = torch.clamp(S_T - K_2d, min=0.0)
        return payoffs

    def train(
        self,
        num_batches: int,
        batch_size: int,
        learning_rate: float = 1e-3,
    ) -> None:
        """Train the CVNN over multiple Sobol-sampled mini-batches.

        Args:
            num_batches: Number of training iterations (each iteration uses fresh Sobol points).
            batch_size: How many parameter sets per iteration.
            learning_rate: Adam learning rate.
        """
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        for step in range(1, num_batches + 1):
            params_unit = self.sobol_engine.draw(
                batch_size
            )  # shape (batch_size, 6), CPU
            # Map to domain
            params_actual = self.domain_lower + (
                self.domain_upper - self.domain_lower
            ) * params_unit.to(self.device)

            payoffs = self._simulate_payoffs(
                params_actual
            )  # (network_size, batch_size)
            payoff_dft = torch.fft.fft(
                payoffs, dim=0
            )  # shape (network_size, batch_size), complex
            target: Tensor = payoff_dft.T  # (batch_size, network_size)

            # Input to model is the normalized param (i.e. params_unit) in [0,1]
            inputs: Tensor = params_unit.to(self.device)  # shape (batch_size, 6)
            preds: Tensor = self.model(inputs)  # (batch_size, network_size), complex

            diff = preds - target
            loss = torch.mean(torch.abs(diff) ** 2)

            optimizer.zero_grad()
            loss.backward()  # type: ignore[no-untyped-call]
            optimizer.step()

            if step % 100 == 0 or step == num_batches:
                print(f"Step {step}/{num_batches}, loss = {loss.item():.4e}")

    def predict_option_price(
        self, X0: float, K: float, T: float, r: float, d: float, v: float
    ) -> float:
        """Use the trained model to predict the mean payoff (DFT 0th frequency / network_size).

        Args:
            X0: Underlying spot price.
            K: Strike price.
            T: Time to maturity (years).
            r: Risk-free rate.
            d: Continuous dividend yield.
            v: Volatility (std dev).

        Returns:
            A float representing the predicted undiscounted mean payoff.
        """
        self.model.eval()
        param_arr = torch.tensor(
            [X0, K, T, r, d, v], dtype=torch.float32, device=self.device
        )
        # Normalize
        param_unit: Tensor = (param_arr - self.domain_lower) / (
            self.domain_upper - self.domain_lower
        )
        param_unit = param_unit.unsqueeze(0)  # shape (1, 6)
        with torch.no_grad():
            pred_dft: Tensor = self.model(param_unit)
        # The 0-frequency component is pred_dft[0, 0]
        payoff_mean = (pred_dft[0, 0].real / float(self.network_size)).item()
        return float(payoff_mean)


if __name__ == "__main__":
    # Example domain (X0, K, T, r, d, v)
    domain_example = {
        "X0": (50.0, 150.0),
        "K": (50.0, 150.0),
        "T": (0.1, 2.0),
        "r": (0.0, 0.1),
        "d": (0.0, 0.05),
        "v": (0.1, 0.5),
    }

    # Instantiate trainer
    trainer = OptionPricerTrainer(
        domain_bounds=domain_example,
        network_size=64,
        hidden_features=32,
        num_res_layers=1,
        scramble=True,
        sobol_seed=42,
        device=None,  # will auto-select (GPU if supported, else CPU)
        dtype=torch.float32,
    )

    # Train with 200 batches of size 16 each
    trainer.train(num_batches=200, batch_size=16, learning_rate=1e-3)

    # Predict for a specific parameter set
    price_pred = trainer.predict_option_price(
        X0=100.0, K=100.0, T=1.0, r=0.02, d=0.01, v=0.2
    )
    print(f"Predicted payoff mean (undiscounted): {price_pred:.4f}")
