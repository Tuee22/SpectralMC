# src/spectralmc/gbm_trainer.py
"""Training infrastructure for learning discounted Black-Scholes pay-off
spectra with a complex-valued neural network (CVNN).

The trainer

* drives Monte-Carlo pricing of options entirely on **CUDA**,
* performs an FFT of the simulated pay-off distribution,
* feeds the complex spectrum into a **PyTorch** CVNN,
* back-propagates mean-squared error against the true spectrum, and
* logs/serialises everything deterministically.

Networks are supplied externally; they must merely satisfy the
:class:`ComplexValuedModel` protocol.  A convenient way to create such a
model is :pyfunc:`spectralmc.cvnn_factory.build_model`.

Note
----
Unlike the original implementation, the trainer now **asserts** that the
supplied model is *already* on the expected ``device``/**real** ``dtype``.
This is done via :pyfunc:`assert_param_attr`, a single-expression helper that
passes ``mypy --strict`` while avoiding imperative statements.
"""

from __future__ import annotations

import math
import time
import warnings
from typing import (
    Callable,
    Dict,
    Iterable,
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
    "GbmTrainerConfig",
    "StepMetrics",
    "TensorBoardLogger",
    "GbmTrainer",
)

# =============================================================================
#  Typing helpers
# =============================================================================


@runtime_checkable
class ComplexValuedModel(Protocol):
    """Callable *complex* network: ``(real, imag) -> (real, imag)``."""

    # forward pass ---------------------------------------------------------
    def __call__(
        self, __real: torch.Tensor, __imag: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]: ...

    # subset of nn.Module API used by the trainer -------------------------
    def parameters(self) -> Iterable[nn.Parameter]: ...
    def named_parameters(self) -> Iterable[Tuple[str, nn.Parameter]]: ...
    def to(self, device: torch.device, dtype: torch.dtype) -> nn.Module: ...
    def train(self, mode: bool = True) -> None: ...
    def eval(self) -> None: ...


# =============================================================================
#  Generic assertion helper (expression-only)
# =============================================================================

_T = TypeVar("_T")


def assert_param_attr(
    model: ComplexValuedModel,
    *,
    attr: str,
    expected: _T,
) -> None:
    """Fail fast if **any** parameter's ``attr`` ≠ ``expected``.

    Implemented without statements—only expressions—so it satisfies the
    user's style constraint and ``mypy --strict`` type-checks.
    """
    (
        lambda mismatches: mismatches
        and (_ for _ in ()).throw(  # expression-style raise
            RuntimeError(
                f"Model parameters violate required {attr}:\n  "
                + "\n  ".join(mismatches)
            )
        )
    )(
        [
            f"{name}: {attr}={getattr(p, attr)} (expected {expected})"
            for name, p in model.named_parameters()
            if getattr(p, attr) != expected
        ]
    )


# =============================================================================
#  Public configuration / metric dataclasses
# =============================================================================

StepLogger = Callable[["StepMetrics"], None]
"""Hook executed after every optimiser step."""


class GbmTrainerConfig(BaseModel):
    """Frozen state for snapshot/restore of :class:`GbmTrainer`."""

    cfg: BlackScholesConfig
    domain_bounds: Dict[str, BoundSpec]
    cvnn: ComplexValuedModel
    optimizer_state: Optional[AdamOptimizerState] = None
    global_step: int = 0

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)


class StepMetrics(BaseModel):
    """Scalar diagnostics emitted after one optimisation step."""

    step: int
    batch_time: float
    loss: float
    grad_norm: float
    lr: float
    optimizer: optim.Adam
    model: ComplexValuedModel

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)


# =============================================================================
#  Logging helpers
# =============================================================================


class TensorBoardLogger:
    """Log :class:`StepMetrics` to TensorBoard."""

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

    def __call__(self, metrics: StepMetrics) -> None:  # noqa: D401
        """Write one step of metrics."""
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
        """Close the underlying writer."""
        self._writer.close()


# =============================================================================
#  Trainer
# =============================================================================


class GbmTrainer:
    """Coordinates MC pricing, FFT and CVNN optimisation."""

    # ------------------------------------------------------------------ #
    # Construction / checkpoint                                          #
    # ------------------------------------------------------------------ #

    def __init__(self, cfg: GbmTrainerConfig) -> None:
        """Instantiate from an immutable configuration."""
        self._sim_params = cfg.cfg.sim_params
        self._cvnn: ComplexValuedModel = cfg.cvnn
        self._domain_bounds = cfg.domain_bounds
        self._optimizer_state = cfg.optimizer_state
        self._global_step = cfg.global_step

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._torch_rdtype = (
            torch.float32 if self._sim_params.dtype == "float32" else torch.float64
        )

        # --- assert model already configured as required ---------------
        assert_param_attr(self._cvnn, attr="device", expected=self._device)
        assert_param_attr(self._cvnn, attr="dtype", expected=self._torch_rdtype)

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
            CuPyStream(non_blocking=False) if self._device.type == "cuda" else None
        )

        self._mc_engine = BlackScholes(cfg.cfg) if self._device.type == "cuda" else None

        self._sampler: SobolSampler[BlackScholes.Inputs] = SobolSampler(
            pydantic_class=BlackScholes.Inputs,
            dimensions=self._domain_bounds,
            skip=self._sim_params.skip,
            seed=self._sim_params.mc_seed,
        )

    # ------------------------------------------------------------------ #
    # Public checkpoint interface                                        #
    # ------------------------------------------------------------------ #

    def snapshot(self) -> GbmTrainerConfig:
        """Return an *immutable* copy of all trainer state."""
        if self._device.type != "cuda" or self._mc_engine is None:
            raise RuntimeError("Snapshots can only be taken on CUDA.")

        return GbmTrainerConfig(
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
        """MC-simulate one contract and return the mean FFT."""
        if self._device.type != "cuda" or self._mc_engine is None:
            raise RuntimeError("Simulation requires CUDA.")

        prices = self._mc_engine.price(inputs=contract).put_price
        price_mat = prices.reshape(
            self._sim_params.batches_per_mc_run, self._sim_params.network_size
        )
        return cp.mean(cp.fft.fft(price_mat, axis=1), axis=0)

    def _torch_step(
        self,
        real_in: torch.Tensor,
        imag_in: torch.Tensor,
        targets: torch.Tensor,
        optimizer: optim.Optimizer,
    ) -> Tuple[torch.Tensor, float]:
        """One forward/backward/optimiser step."""
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
    # Public API                                                         #
    # ------------------------------------------------------------------ #

    def train(
        self,
        *,
        num_batches: int,
        batch_size: int,
        learning_rate: float,
        logger: Optional[StepLogger] = None,
    ) -> None:
        """Run *num_batches* optimisation steps."""
        if self._device.type != "cuda":
            raise RuntimeError("Training requires a CUDA-capable GPU.")

        adam = optim.Adam(self._cvnn.parameters(), lr=learning_rate)
        if self._optimizer_state is not None:
            adam.load_state_dict(self._optimizer_state.to_torch(device=self._device))

        self._cvnn.train()

        for _ in range(num_batches):
            tic = time.perf_counter()

            # ---------------- Monte-Carlo & FFT (CuPy) ----------------
            sobol_inputs = self._sampler.sample(batch_size)
            assert self._cupy_stream is not None  # mypy
            with self._cupy_stream:
                fft_buf = cp.asarray(
                    [self._simulate_fft(contract) for contract in sobol_inputs],
                    dtype=self._cupy_cdtype,
                )
            self._cupy_stream.synchronize()

            # ---------------- CVNN forward/backward (Torch) ----------
            assert self._torch_stream is not None  # mypy
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

            # ---------------- Logging --------------------------------
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

        self._optimizer_state = AdamOptimizerState.from_torch(adam)

    # ------------------------------------------------------------------ #
    # Inference                                                          #
    # ------------------------------------------------------------------ #

    def predict_price(
        self, inputs: Sequence[BlackScholes.Inputs]
    ) -> List[BlackScholes.HostPricingResults]:
        """Vectorised pricing of plain-vanilla European options."""
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


# =============================================================================
#  Pure helpers
# =============================================================================


def _split_inputs(
    inputs: Sequence[BlackScholes.Inputs], *, dtype: torch.dtype, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert Pydantic inputs into (real, imag) tensors."""
    fields = list(BlackScholes.Inputs.model_fields.keys())
    rows = [[float(getattr(inp, f)) for f in fields] for inp in inputs]
    real = torch.tensor(rows, dtype=dtype, device=device)
    imag = torch.zeros_like(real)
    return real, imag
