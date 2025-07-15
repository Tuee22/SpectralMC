# src/spectralmc/gbm_trainer.py
"""
CUDA‑accelerated trainer for learning discounted Black–Scholes pay‑off
spectra with a complex‑valued neural network (CVNN).

Overview
--------
The :class:`GbmCVNNPricer` orchestrates four tightly‑coupled components:

1. **Quasi‑random contract generator** – `SobolSampler` draws input
   contracts inside user‑supplied bounds.
2. **Monte‑Carlo engine** – `spectralmc.gbm.BlackScholes` prices a batch
   of contracts entirely on **CUDA**, then applies an FFT along the time
   axis.
3. **Complex‑valued neural network** – any user model that satisfies the
   :class:`ComplexValuedModel` protocol.  The trainer infers its
   execution *device* and *real dtype* directly from the network’s first
   parameter and asserts that **all** parameters conform.
4. **Optimiser & logging** – an *Adam* optimiser operating in
   half‑open‑loop: model parameters live on the active CUDA device while
   the optimiser *state* is moved to CPU **before** every snapshot to
   satisfy :pyclass:`spectralmc.models.torch.AdamOptimizerState`.

Determinism
-----------
*   Global PRNG seeding is delegated to callers.
*   Model and optimiser states are serialised via pure‑CPU
    :pyclass:`~spectralmc.models.torch.TensorState` objects; this avoids
    device‑specific artefacts in unit tests.

Implementation notes
--------------------
* ``_move_optimizer_state`` is a tiny helper that migrates *all* tensors
  in an `torch.optim.Optimizer.state` dict between devices.
* The trainer *always* stores the optimiser snapshot on **CPU**.  During
  a restart we immediately migrate the state back to the training
  device.
* Strict typing is enforced with **mypy --strict**; no `Any`, `cast` or
  `type: ignore` are required.
"""

from __future__ import annotations

import math
import time
import warnings
from typing import (
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    TypeVar,
    runtime_checkable,
)

import cupy as cp
from cupy.cuda import Stream as CuPyStream
import torch
import torch.nn as nn
import torch.optim as optim
from pydantic import BaseModel, ConfigDict
from torch.utils.tensorboard import SummaryWriter

from spectralmc.gbm import BlackScholes, BlackScholesConfig
from spectralmc.models.torch import AdamOptimizerState
from spectralmc.sobol_sampler import BoundSpec, SobolSampler

__all__: Tuple[str, ...] = (
    "GbmCVNNPricerConfig",
    "StepMetrics",
    "TensorBoardLogger",
    "GbmCVNNPricer",
)

# =============================================================================
# Protocols & helpers
# =============================================================================


@runtime_checkable
class ComplexValuedModel(Protocol):
    """Callable complex‑valued network ``(real, imag) -> (real, imag)``."""

    def __call__(
        self, __real: torch.Tensor, __imag: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]: ...

    # minimal subset of the nn.Module API
    def parameters(self) -> Iterable[nn.Parameter]: ...
    def named_parameters(self) -> Iterable[Tuple[str, nn.Parameter]]: ...
    def to(self, device: torch.device, dtype: torch.dtype) -> nn.Module: ...
    def train(self, mode: bool = True) -> None: ...
    def eval(self) -> None: ...


# def assert_param_attr(model: ComplexValuedModel, *, attr: str, expected: _T) -> None:
#     """Fail if *any* parameter’s ``attr`` differs from *expected*."""
#     mismatches = [
#         f"{name}: {attr}={getattr(p, attr)} (expected {expected})"
#         for name, p in model.named_parameters()
#         if getattr(p, attr) != expected
#     ]
#     if mismatches:
#         raise RuntimeError(
#             f"Model parameters violate required {attr}:\n  " + "\n  ".join(mismatches)
#         )


# =============================================================================
# Immutable dataclasses
# =============================================================================

StepLogger = Callable[["StepMetrics"], None]
"""User‑supplied callback executed once after every optimiser step."""


class GbmCVNNPricerConfig(BaseModel):
    """Frozen snapshot used for (de‑)serialising a trainer."""

    cfg: BlackScholesConfig
    domain_bounds: Dict[str, BoundSpec]
    cvnn: ComplexValuedModel
    optimizer_state: Optional[AdamOptimizerState] = None
    global_step: int = 0

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)


class StepMetrics(BaseModel):
    """Scalar diagnostics emitted at the end of every optimiser step."""

    step: int
    batch_time: float
    loss: float
    grad_norm: float
    lr: float
    optimizer: optim.Adam
    model: ComplexValuedModel

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)


# =============================================================================
#  TensorBoard logger
# =============================================================================


class TensorBoardLogger:
    """Light‑weight TensorBoard logger for :class:`StepMetrics`."""

    def __init__(
        self,
        *,
        logdir: str = ".logs/gbm_trainer",
        hist_every: int = 10,
        flush_every: int = 100,
    ) -> None:
        self._writer = SummaryWriter(log_dir=logdir)
        self._hist_every = max(hist_every, 1)
        self._flush_every = max(flush_every, 1)

    # ------------------------------------------------------------------ #
    # Callable interface                                                 #
    # ------------------------------------------------------------------ #

    def __call__(self, metrics: StepMetrics) -> None:
        """Write one optimisation step to disk."""
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
        """Close the underlying :class:`~torch.utils.tensorboard.SummaryWriter`."""
        self._writer.close()


# =============================================================================
#  Trainer
# =============================================================================


class GbmCVNNPricer:
    """Coordinate MC pricing, FFT and CVNN optimisation."""

    # ------------------------------------------------------------------ #
    # Construction                                                       #
    # ------------------------------------------------------------------ #

    def __init__(self, cfg: GbmCVNNPricerConfig) -> None:
        """Instantiate a trainer from its immutable configuration."""
        # User payload -------------------------------------------------- #
        self._sim_params = cfg.cfg.sim_params
        self._cvnn: ComplexValuedModel = cfg.cvnn
        self._domain_bounds = cfg.domain_bounds
        self._optimizer_state = cfg.optimizer_state
        self._global_step = cfg.global_step

        # Inferred execution device ------------------------------------ #
        first_param = next(iter(self._cvnn.parameters()), None)
        if first_param is None:
            raise RuntimeError("Model has no parameters – cannot infer device.")
        self._device: torch.device = first_param.device
        assert_param_attr(self._cvnn, attr="device", expected=self._device)

        # Scalar dtype checks ------------------------------------------- #
        self._torch_rdtype = (
            torch.float32 if self._sim_params.dtype == "float32" else torch.float64
        )
        assert_param_attr(self._cvnn, attr="dtype", expected=self._torch_rdtype)

        # Complex dtypes & streams -------------------------------------- #
        self._torch_cdtype = (
            torch.complex64 if self._sim_params.dtype == "float32" else torch.complex128
        )
        self._cupy_cdtype = (
            cp.complex64 if self._sim_params.dtype == "float32" else cp.complex128
        )

        self._torch_stream: Optional[torch.cuda.Stream] = (
            torch.cuda.Stream(device=self._device)
            if self._device.type == "cuda"
            else None
        )
        self._cupy_stream: Optional[CuPyStream] = (
            CuPyStream() if self._device.type == "cuda" else None
        )

        # Engines -------------------------------------------------------- #
        self._mc_engine = BlackScholes(cfg.cfg) if self._device.type == "cuda" else None
        self._sampler: SobolSampler[BlackScholes.Inputs] = SobolSampler(
            pydantic_class=BlackScholes.Inputs,
            dimensions=self._domain_bounds,
            skip=self._sim_params.skip,
            seed=self._sim_params.mc_seed,
        )

    # ------------------------------------------------------------------ #
    # Checkpointing                                                      #
    # ------------------------------------------------------------------ #

    def snapshot(self) -> GbmCVNNPricerConfig:
        """
        Return a fully‑deterministic snapshot.

        Note
        ----
        The optimiser state is **moved to CPU** so that the resulting
        :class:`AdamOptimizerState` contains only CPU tensors – a hard
        requirement for the strict serialisation helper.
        """
        if self._device.type != "cuda" or self._mc_engine is None:
            raise RuntimeError("Snapshots can only be taken on CUDA.")
        return GbmCVNNPricerConfig(
            cfg=self._mc_engine.snapshot(),
            domain_bounds=self._domain_bounds,
            cvnn=self._cvnn,
            optimizer_state=self._optimizer_state,
            global_step=self._global_step,
        )

    # ------------------------------------------------------------------ #
    # Internal helpers                                                   #
    # ------------------------------------------------------------------ #

    def _simulate_fft(self, contract: BlackScholes.Inputs) -> cp.ndarray:
        """Simulate one contract and return the batch‑mean FFT."""
        if self._mc_engine is None:
            raise RuntimeError("Monte‑Carlo engine not initialised.")
        prices = self._mc_engine.price(inputs=contract).put_price
        mat = prices.reshape(
            self._sim_params.batches_per_mc_run, self._sim_params.network_size
        )
        return cp.mean(cp.fft.fft(mat, axis=1), axis=0)

    def _torch_step(
        self,
        real_in: torch.Tensor,
        imag_in: torch.Tensor,
        targets: torch.Tensor,
        optimizer: optim.Optimizer,
    ) -> Tuple[torch.Tensor, float]:
        """One forward/backward/optimiser step; returns ``(loss, grad_norm)``."""
        pred_r, pred_i = self._cvnn(real_in, imag_in)
        loss = nn.functional.mse_loss(
            pred_r, torch.real(targets)
        ) + nn.functional.mse_loss(pred_i, torch.imag(targets))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        grad_norm = float(
            torch.nn.utils.clip_grad_norm_(self._cvnn.parameters(), float("inf"))
        )
        return loss, grad_norm

    # ------------------------------------------------------------------ #
    # Public API: training                                               #
    # ------------------------------------------------------------------ #

    def train(
        self,
        *,
        num_batches: int,
        batch_size: int,
        learning_rate: float,
        logger: Optional[StepLogger] = None,
    ) -> None:
        """Run **CUDA‑only** optimisation for *num_batches* steps."""
        if self._device.type != "cuda":
            raise RuntimeError("Trainer requires a CUDA‑capable GPU.")

        adam = optim.Adam(self._cvnn.parameters(), lr=learning_rate)

        # (re‑)attach previous optimiser state -------------------------- #
        if self._optimizer_state is not None:
            adam.load_state_dict(self._optimizer_state.to_torch())
            _move_optimizer_state(adam, self._device)  # GPU ← CPU

        self._cvnn.train()

        # main loop ------------------------------------------------------ #
        for _ in range(num_batches):
            tic = time.perf_counter()

            # ── Monte‑Carlo + FFT (CuPy) ─────────────────────────────── #
            sobol_inputs = self._sampler.sample(batch_size)
            assert self._cupy_stream is not None
            with self._cupy_stream:
                fft_buf = cp.asarray(
                    [self._simulate_fft(c) for c in sobol_inputs],
                    dtype=self._cupy_cdtype,
                )
            self._cupy_stream.synchronize()

            # ── CVNN step (Torch) ────────────────────────────────────── #
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
                loss, grad_norm = self._torch_step(real_in, imag_in, targets, adam)
            self._torch_stream.synchronize()

            # ── Logging ──────────────────────────────────────────────── #
            if logger is not None:
                logger(
                    StepMetrics(
                        step=self._global_step,
                        batch_time=time.perf_counter() - tic,
                        loss=loss.item(),
                        grad_norm=grad_norm,
                        lr=float(adam.param_groups[0]["lr"]),
                        optimizer=adam,
                        model=self._cvnn,
                    )
                )
            self._global_step += 1

        # ── Snapshot optimiser (CPU) ─────────────────────────────────── #
        _move_optimizer_state(adam, torch.device("cpu"))  # CPU ← GPU
        self._optimizer_state = AdamOptimizerState.from_torch(adam)

    # ------------------------------------------------------------------ #
    # Public API: inference                                              #
    # ------------------------------------------------------------------ #

    def predict_price(
        self, inputs: Sequence[BlackScholes.Inputs]
    ) -> List[BlackScholes.HostPricingResults]:
        """Vectorised valuation of plain‑vanilla European options."""
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

        spectrum = torch.view_as_complex(torch.stack((pred_r, pred_i), dim=-1))
        avg_ifft = torch.fft.ifft(spectrum, dim=1).mean(dim=1)

        results: List[BlackScholes.HostPricingResults] = []
        for coeff, contract in zip(avg_ifft, inputs, strict=True):
            real_val = float(torch.real(coeff).item())
            imag_val = float(torch.imag(coeff).item())
            if abs(imag_val) > 1.0e-6:
                warnings.warn(
                    f"IFFT imaginary component {imag_val:.3e} exceeds tolerance.",
                    RuntimeWarning,
                )

            discount = math.exp(-contract.r * contract.T)
            forward = contract.X0 * math.exp((contract.r - contract.d) * contract.T)
            put_price = real_val
            call_price = put_price + forward - contract.K * discount

            results.append(
                BlackScholes.HostPricingResults(
                    underlying=forward,
                    put_price=put_price,
                    call_price=call_price,
                    put_price_intrinsic=discount * max(contract.K - forward, 0.0),
                    call_price_intrinsic=discount * max(forward - contract.K, 0.0),
                    put_convexity=put_price - discount * max(contract.K - forward, 0.0),
                    call_convexity=call_price
                    - discount * max(forward - contract.K, 0.0),
                )
            )
        return results


# =============================================================================
#  Pure helpers
# =============================================================================


def _split_inputs(
    inputs: Sequence[BlackScholes.Inputs], *, dtype: torch.dtype, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert Pydantic input contracts into ``(real, imag)`` tensors."""
    fields = list(BlackScholes.Inputs.model_fields.keys())
    rows = [[float(getattr(inp, f)) for f in fields] for inp in inputs]
    real = torch.tensor(rows, dtype=dtype, device=device)
    imag = torch.zeros_like(real)
    return real, imag
