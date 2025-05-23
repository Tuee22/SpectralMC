# spectralmc/gbm_trainer.py
# mypy: disable-error-code=no-untyped-call

"""
Deterministic CVNN trainer with append-only S3/MinIO checkpoints.

Key points
----------
* Single `global_seed` → deterministic NumPy RNG → seeds for torch & Sobol.
* Trainer state (tensors + BlackScholesConfig) is saved immutably under a
  new key each time `train()` completes.
* `predict_price()` preserved for inference/testing.
* Fully strict-typed (`mypy --strict` passes with zero errors/warnings).
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import random
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Tuple

import boto3
import cupy as cp  # type: ignore[import-untyped]
import numpy as np
import torch
import torch.nn as nn
from pydantic import BaseModel
from torch.utils.tensorboard import SummaryWriter

from spectralmc.cvnn import CVNN
from spectralmc.gbm import BlackScholes, BlackScholesConfig
from spectralmc.sobol_sampler import BoundSpec, SobolSampler

# --------------------------------------------------------------------------- #
# Constants                                                                   #
# --------------------------------------------------------------------------- #

IMAG_TOL: float = 1e-6
DEFAULT_LEARNING_RATE: float = 1e-3
GLOBAL_SEED_DEFAULT: int = 42
TRAIN_LOG_INTERVAL: int = 10
LOG_DIR_DEFAULT: str = "s3://opt-models/tb"

S3_BUCKET: str = "opt-models"
S3_PREFIX: str = "cvnn/v1"

# --------------------------------------------------------------------------- #
# Utilities                                                                   #
# --------------------------------------------------------------------------- #


def _torch_dtype(sim_precision: str) -> torch.dtype:
    return torch.float32 if sim_precision == "float32" else torch.float64


def _inputs_to_real_imag(
    inputs: List[BlackScholes.Inputs], dtype: torch.dtype, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    names = list(BlackScholes.Inputs.model_fields.keys())
    rows = [[float(getattr(inp, n)) for n in names] for inp in inputs]
    real = torch.tensor(rows, dtype=dtype, device=device)
    imag = torch.zeros_like(real)
    return real, imag


# --------------------------------------------------------------------------- #
# TensorBoard façade                                                          #
# --------------------------------------------------------------------------- #


class _TBLogger:
    def __init__(self, logdir: str, hist_every: int, flush_every: int) -> None:
        self._writer = SummaryWriter(log_dir=logdir, flush_secs=flush_every)
        self._hist_every = max(1, hist_every)

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
            for name, p in model.named_parameters():
                w.add_histogram(name, p, step)
                if p.grad is not None:
                    w.add_histogram(f"{name}.grad", p.grad, step)

    def close(self) -> None:
        self._writer.flush()
        self._writer.close()


# --------------------------------------------------------------------------- #
# Metadata model                                                              #
# --------------------------------------------------------------------------- #


class _Meta(BaseModel):
    version_id: str
    parent_version: str | None
    step: int
    ckpt_key: str
    created_at: str
    bs_cfg: BlackScholesConfig


# --------------------------------------------------------------------------- #
# Trainer                                                                     #
# --------------------------------------------------------------------------- #


class GbmTrainer:
    """Reproducible GBM-MC + CVNN trainer with append-only checkpoints."""

    # ---------------- constructor -------------------------------------- #

    def __init__(
        self,
        *,
        cfg: BlackScholesConfig,
        domain_bounds: dict[str, BoundSpec],
        cvnn: CVNN,
        device: torch.device | None = None,
        tb_logdir: str = LOG_DIR_DEFAULT,
        hist_every: int = 10,
        flush_every: int = 100,
        global_seed: int = GLOBAL_SEED_DEFAULT,
    ) -> None:
        # ---------- deterministic seeding ------------------------------ #
        base_rng = np.random.default_rng(global_seed)
        torch_seed = int(base_rng.integers(0, 2**63 - 1))
        sobol_seed = int(base_rng.integers(0, 2**63 - 1))

        random.seed(global_seed)
        np.random.seed(global_seed)
        torch.manual_seed(torch_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(torch_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # ---------- device & model ------------------------------------- #
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        self.cfg = cfg
        self.cvnn = cvnn.to(device=device, dtype=_torch_dtype(cfg.sim_params.dtype))

        self.torch_stream: torch.cuda.Stream | None = None
        if self.device.type == "cuda":
            self.torch_stream = torch.cuda.Stream(device=self.device)

        # ---------- Sobol sampler & MC engine --------------------------- #
        self.sampler = SobolSampler(
            pydantic_class=BlackScholes.Inputs,
            dimensions=domain_bounds,
            skip=cfg.sim_params.skip,
            seed=sobol_seed,
        )
        self.mc_engine = BlackScholes(cfg)

        # ---------- logging & storage ---------------------------------- #
        self._tb = _TBLogger(tb_logdir, hist_every, flush_every)
        self._s3 = boto3.client("s3", endpoint_url=os.getenv("AWS_ENDPOINT_URL"))
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self._run_prefix = f"{S3_PREFIX}/runs/{ts}"
        self._parent_version: str | None = None

    # ---------------- training loop ----------------------------------- #

    def train(
        self,
        *,
        num_batches: int,
        batch_size: int,
        learning_rate: float = DEFAULT_LEARNING_RATE,
    ) -> None:
        if self.device.type != "cuda":
            raise RuntimeError("CUDA device required.")
        assert self.torch_stream is not None  # for mypy

        optimiser = torch.optim.Adam(self.cvnn.parameters(), lr=learning_rate)
        self.cvnn.train()

        cp_dtype = (
            cp.complex64 if self.cfg.sim_params.dtype == "float32" else cp.complex128
        )
        torch_cdtype = (
            torch.complex64
            if self.cfg.sim_params.dtype == "float32"
            else torch.complex128
        )
        torch_rdtype = _torch_dtype(self.cfg.sim_params.dtype)

        global_step = 0
        for step in range(1, num_batches + 1):
            t0 = time.perf_counter()

            sobol_pts = self.sampler.sample(batch_size)
            fft_buf = cp.zeros(
                (batch_size, self.cfg.sim_params.network_size), dtype=cp_dtype
            )

            for i, contract in enumerate(sobol_pts):
                mc = self.mc_engine.price(inputs=contract)
                puts = mc.put_price.reshape(
                    (
                        self.cfg.sim_params.batches_per_mc_run,
                        self.cfg.sim_params.network_size,
                    )
                )
                fft_buf[i, :] = cp.mean(cp.fft.fft(puts, axis=1), axis=0)

            cp.cuda.Stream.null.synchronize()

            with torch.cuda.stream(self.torch_stream):
                targets = torch.utils.dlpack.from_dlpack(fft_buf.toDlpack()).to(
                    torch_cdtype
                )
                r_in, i_in = _inputs_to_real_imag(sobol_pts, torch_rdtype, self.device)
                pred_r, pred_i = self.cvnn(r_in, i_in)
                loss = nn.functional.mse_loss(
                    pred_r, targets.real
                ) + nn.functional.mse_loss(pred_i, targets.imag)

                optimiser.zero_grad(set_to_none=True)
                loss.backward()
                optimiser.step()

                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.cvnn.parameters(), max_norm=float("inf")
                ).item()

            self.torch_stream.synchronize()
            batch_time = time.perf_counter() - t0
            lr = optimiser.param_groups[0]["lr"]

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
                    f"[TRAIN] {step}/{num_batches}  "
                    f"loss={loss.item():.4g}  "
                    f"{batch_time*1e3:6.1f} ms"
                )

        self._tb.close()
        self._commit_checkpoint(global_step, optimiser)

    # ---------------- inference --------------------------------------- #

    def predict_price(
        self, inputs: List[BlackScholes.Inputs]
    ) -> List[BlackScholes.HostPricingResults]:
        if not inputs:
            return []

        self.cvnn.eval()
        rdtype = _torch_dtype(self.cfg.sim_params.dtype)
        real_in, imag_in = _inputs_to_real_imag(inputs, rdtype, self.device)

        if self.torch_stream is None:
            with torch.no_grad():
                pred_r, pred_i = self.cvnn(real_in, imag_in)
        else:
            with torch.no_grad(), torch.cuda.stream(self.torch_stream):
                pred_r, pred_i = self.cvnn(real_in, imag_in)
            self.torch_stream.synchronize()

        spectrum = torch.complex(pred_r, pred_i)
        mean_ifft = torch.fft.ifft(spectrum, dim=1).mean(dim=1)

        results: List[BlackScholes.HostPricingResults] = []
        for cval, bs in zip(mean_ifft, inputs):
            real_val = float(torch.real(cval).item())
            imag_val = float(torch.imag(cval).item())
            if abs(imag_val) > IMAG_TOL:
                warnings.warn(
                    f"IFFT imaginary part {imag_val:.3e} exceeds tolerance",
                    RuntimeWarning,
                )

            disc = np.exp(-bs.r * bs.T)
            fwd = bs.X0 * np.exp((bs.r - bs.d) * bs.T)

            put_intr = disc * max(bs.K - fwd, 0.0)
            call_val = real_val + fwd - bs.K * disc
            call_intr = disc * max(fwd - bs.K, 0.0)

            results.append(
                BlackScholes.HostPricingResults(
                    put_price_intrinsic=put_intr,
                    call_price_intrinsic=call_intr,
                    underlying=fwd,
                    put_convexity=real_val - put_intr,
                    call_convexity=call_val - call_intr,
                    put_price=real_val,
                    call_price=call_val,
                )
            )
        return results

    # ---------------- checkpoint upload -------------------------------- #

    def _commit_checkpoint(self, step: int, opt: torch.optim.Optimizer) -> None:
        lock_key = f"{self._run_prefix}/LOCK"
        try:
            self._s3.put_object(
                Bucket=S3_BUCKET, Key=lock_key, Body=b"", IfNoneMatch="*"
            )
        except self._s3.exceptions.ClientError as e:
            raise RuntimeError("Another writer is active") from e

        try:
            buf = io.BytesIO()
            torch.save(
                {
                    "model": self.cvnn.state_dict(),
                    "optim": opt.state_dict(),
                    "step": step,
                },
                buf,
            )
            payload = buf.getvalue()
            sha = hashlib.sha256(payload).hexdigest()
            ckpt_key = f"{self._run_prefix}/ckpt-{step}-{sha[:8]}.pt"
            self._s3.put_object(Bucket=S3_BUCKET, Key=ckpt_key, Body=payload)

            meta = _Meta(
                version_id=sha,
                parent_version=self._parent_version,
                step=step,
                ckpt_key=ckpt_key,
                created_at=datetime.now(timezone.utc).isoformat(),
                bs_cfg=self.cfg,
            )
            meta_key = f"{self._run_prefix}/meta-{step}-{sha}.json"
            self._s3.put_object(
                Bucket=S3_BUCKET,
                Key=meta_key,
                Body=json.dumps(meta.model_dump(), indent=2).encode(),
            )
            self._s3.put_object(
                Bucket=S3_BUCKET, Key=f"{self._run_prefix}/HEAD", Body=meta_key.encode()
            )
            self._parent_version = sha
        finally:
            self._s3.delete_object(Bucket=S3_BUCKET, Key=lock_key)

    # ---------------- restore helper ----------------------------------- #

    @classmethod
    def load_from_uri(
        cls, uri: str, *, device: torch.device | None = None
    ) -> "GbmTrainer":
        if uri.startswith("s3://"):
            bucket, key = uri[5:].split("/", 1)
            s3 = boto3.client("s3", endpoint_url=os.getenv("AWS_ENDPOINT_URL"))
            meta_json = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
            meta = _Meta.model_validate_json(meta_json)
            ckpt_bytes = s3.get_object(Bucket=bucket, Key=meta.ckpt_key)["Body"].read()
        else:
            with Path(uri).open("rb") as fh:
                meta = _Meta.model_validate_json(fh.read())
            with Path(meta.ckpt_key).open("rb") as fh:
                ckpt_bytes = fh.read()

        ckpt = torch.load(io.BytesIO(ckpt_bytes), map_location="cpu")
        cvnn = CVNN(
            input_features=6,
            output_features=meta.bs_cfg.sim_params.network_size,
            hidden_features=meta.bs_cfg.sim_params.network_size * 2,
            num_residual_blocks=2,
        )
        cvnn.load_state_dict(ckpt["model"])

        return cls(
            cfg=meta.bs_cfg,
            domain_bounds={},  # supply the original bounds here
            cvnn=cvnn,
            device=device,
            global_seed=GLOBAL_SEED_DEFAULT,
        )
