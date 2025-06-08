from __future__ import annotations

"""
GPU trainer that learns the discrete Fourier transform (DFT) of discounted
Black-Scholes pay-off distributions with a complex-valued neural network
(**CVNN**) entirely on CUDA.

Highlights:
    - CUDA-only training
    - Dedicated CUDA streams for FFT and autograd
    - Pluggable per-step logging (e.g. TensorBoard)
    - Deterministic snapshot/restore of state including optimizer buffers
"""

import math
import time
import warnings
from typing import Callable, Dict, List, Mapping, Optional, Tuple, TypeAlias

import cupy as cp
from cupy.cuda import Stream as CuPyStream
import torch
import torch.nn as nn
import torch.optim as optim
from pydantic import BaseModel, ConfigDict
from torch.utils.tensorboard import SummaryWriter

from spectralmc.cvnn import CVNN
from spectralmc.gbm import BlackScholes, BlackScholesConfig
from spectralmc.models.torch import AdamOptimizerState
from spectralmc.sobol_sampler import BoundSpec, SobolSampler

StepLogger: TypeAlias = Callable[["StepMetrics"], None]


class GbmTrainerConfig(BaseModel):
    cfg: BlackScholesConfig
    domain_bounds: Dict[str, BoundSpec]
    cvnn: CVNN
    optimizer_state: Optional[AdamOptimizerState] = None
    global_step: int = 0

    model_config = ConfigDict(arbitrary_types_allowed=True)


class StepMetrics(BaseModel):
    step: int
    batch_time: float
    loss: float
    grad_norm: float
    lr: float
    optimizer: optim.Optimizer
    model: CVNN

    model_config = ConfigDict(arbitrary_types_allowed=True)


class TensorBoardLogger:
    def __init__(
        self,
        *,
        logdir: str = ".logs/gbm_trainer",
        hist_every: int = 10,
        flush_every: int = 100,
    ) -> None:
        self._writer = SummaryWriter(log_dir=logdir)
        self._hist_every = max(1, hist_every)
        self._flush_every = max(1, flush_every)

    def __call__(self, metrics: StepMetrics) -> None:
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

    def close(self) -> None:
        self._writer.close()


class GbmTrainer:
    """Coordinates MC pricing, FFT and CVNN optimisation."""

    def __init__(self, cfg: GbmTrainerConfig) -> None:
        self._sim_params = cfg.cfg.sim_params
        self._cvnn = cfg.cvnn
        self._domain_bounds = cfg.domain_bounds
        self._optimizer_state = cfg.optimizer_state
        self._global_step = cfg.global_step

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._torch_stream: Optional[torch.cuda.Stream] = (
            torch.cuda.Stream(device=self._device)
            if self._device.type == "cuda"
            else None
        )
        self._cupy_stream: Optional[CuPyStream] = (
            CuPyStream(non_blocking=False) if self._device.type == "cuda" else None
        )

        self._sampler: SobolSampler[BlackScholes.Inputs] = SobolSampler(
            pydantic_class=BlackScholes.Inputs,
            dimensions=self._domain_bounds,
            skip=self._sim_params.skip,
            seed=self._sim_params.mc_seed,
        )
        self._mc_engine = BlackScholes(cfg.cfg)

        self._cvnn.to(self._device, _torch_precision_dtype(self._sim_params.dtype))

        self._network_size = self._sim_params.network_size
        self._batches_per_mc_run = self._sim_params.batches_per_mc_run
        self._cp_cdtype = (
            cp.complex64 if self._sim_params.dtype == "float32" else cp.complex128
        )
        self._torch_cdtype = (
            torch.complex64 if self._sim_params.dtype == "float32" else torch.complex128
        )
        self._torch_rdtype = _torch_precision_dtype(self._sim_params.dtype)

    def snapshot(self) -> GbmTrainerConfig:
        return GbmTrainerConfig(
            cfg=self._mc_engine.snapshot(),
            domain_bounds=self._domain_bounds,
            cvnn=self._cvnn,
            optimizer_state=self._optimizer_state,
            global_step=self._global_step,
        )

    def _simulate_fft(self, contract: BlackScholes.Inputs) -> cp.ndarray:
        prices = self._mc_engine.price(inputs=contract).put_price
        price_mat = prices.reshape(self._batches_per_mc_run, self._network_size)
        return cp.mean(cp.fft.fft(price_mat, axis=1), axis=0)

    def _torch_step(
        self,
        real_in: torch.Tensor,
        imag_in: torch.Tensor,
        targets: torch.Tensor,
        optimizer: optim.Optimizer,
    ) -> Tuple[torch.Tensor, float]:
        pred_r, pred_i = self._cvnn(real_in, imag_in)
        loss = nn.functional.mse_loss(pred_r, targets.real) + nn.functional.mse_loss(
            pred_i, targets.imag
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        grad_norm = float(
            torch.nn.utils.clip_grad_norm_(self._cvnn.parameters(), float("inf"))
        )
        return loss, grad_norm

    def train(
        self,
        *,
        num_batches: int,
        batch_size: int,
        learning_rate: float,
        logger: Optional[StepLogger] = None,
    ) -> None:
        if self._device.type != "cuda":
            raise RuntimeError("Training requires a CUDA-capable GPU.")

        adam = optim.Adam(self._cvnn.parameters(), lr=learning_rate)
        if self._optimizer_state is not None:
            adam.load_state_dict(self._optimizer_state.to_torch(device=self._device))

        self._cvnn.train()

        for _ in range(num_batches):
            tic = time.perf_counter()
            sobol_inputs = self._sampler.sample(batch_size)

            assert self._cupy_stream is not None
            with self._cupy_stream:
                fft_buf = cp.asarray(
                    [self._simulate_fft(contract) for contract in sobol_inputs]
                )
            self._cupy_stream.synchronize()

            assert self._torch_stream is not None
            with torch.cuda.stream(self._torch_stream):
                targets = (
                    torch.utils.dlpack.from_dlpack(fft_buf.toDlpack())
                    .to(self._torch_cdtype)
                    .detach()
                )
                real_in, imag_in = _split_inputs(
                    sobol_inputs, dtype=self._torch_rdtype, device=self._device
                )
                loss, grad_norm_val = self._torch_step(real_in, imag_in, targets, adam)
            self._torch_stream.synchronize()

            batch_time = time.perf_counter() - tic
            if logger is not None:
                logger(
                    StepMetrics(
                        step=self._global_step,
                        batch_time=batch_time,
                        loss=loss.item(),
                        grad_norm=grad_norm_val,
                        lr=float(adam.param_groups[0]["lr"]),
                        optimizer=adam,
                        model=self._cvnn,
                    )
                )
            self._global_step += 1

        self._optimizer_state = AdamOptimizerState.from_torch(adam.state_dict())

    def predict_price(
        self, inputs: List[BlackScholes.Inputs]
    ) -> List[BlackScholes.HostPricingResults]:
        if not inputs:
            return []

        self._cvnn.eval()
        real_in, imag_in = _split_inputs(
            inputs, dtype=self._torch_rdtype, device=self._device
        )

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


def _torch_precision_dtype(precision: str) -> torch.dtype:
    return torch.float32 if precision == "float32" else torch.float64


def _split_inputs(
    inputs: List[BlackScholes.Inputs], *, dtype: torch.dtype, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    fields = list(BlackScholes.Inputs.model_fields.keys())
    rows = [[float(getattr(inp, f)) for f in fields] for inp in inputs]
    real = torch.tensor(rows, dtype=dtype, device=device)
    imag = torch.zeros_like(real)
    return real, imag
