# src/spectralmc/gbm_trainer.py
"""Deterministic CVNN trainer with append-only S3 checkpoints."""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import random
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Sequence, Tuple

import boto3
import cupy as cp  # type: ignore[import-untyped]
import numpy as np
import torch
import torch.nn as nn
from pydantic import BaseModel
from torch.utils.tensorboard import SummaryWriter

from spectralmc.cvnn import CVNN, CVNNConfig
from spectralmc.gbm import BlackScholes, BlackScholesConfig
from spectralmc.sobol_sampler import BoundSpec, SobolSampler

IMAG_TOL = 1e-6
DEFAULT_LEARNING_RATE = 1e-3
GLOBAL_SEED_DEFAULT = 42
LOG_DIR_DEFAULT = "s3://opt-models/tb"

S3_BUCKET = "opt-models"
S3_PREFIX = "cvnn/v1"


def _torch_dtype(prec: str) -> torch.dtype:
    return torch.float32 if prec == "float32" else torch.float64


def _inputs_to_real_imag(
    inputs: List[BlackScholes.Inputs], dtype: torch.dtype, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    names = list(BlackScholes.Inputs.model_fields)
    rows = [[float(getattr(inp, n)) for n in names] for inp in inputs]
    real = torch.tensor(rows, dtype=dtype, device=device)
    imag = torch.zeros_like(real)
    return real, imag


def _price_one(engine: BlackScholes, inp: BlackScholes.Inputs) -> Any:
    try:
        return engine.price(inputs=inp)
    except TypeError:
        return engine.price()


class _TBLogger:
    """Minimal wrapper that keeps mypy happy."""

    def __init__(self, logdir: str, hist_every: int, flush_every: int) -> None:
        self._writer: Any = SummaryWriter(log_dir=logdir, flush_secs=flush_every)
        self._hist_every = max(1, hist_every)

    def log_step(
        self,
        model: nn.Module,
        step: int,
        loss: float,
        lr: float,
        grad_norm: float,
        batch_time: float,
    ) -> None:
        w = self._writer
        w.add_scalar("loss/train", loss, step)
        w.add_scalar("lr", lr, step)
        w.add_scalar("grad_norm", grad_norm, step)
        w.add_scalar("batch_time", batch_time, step)

        if step % self._hist_every == 0:
            for name, p in model.named_parameters():
                w.add_histogram(name, p, step)
                if p.grad is not None:
                    w.add_histogram(f"{name}.grad", p.grad, step)

    def close(self) -> None:
        self._writer.flush()
        self._writer.close()


class _Meta(BaseModel):
    version_id: str
    parent_version: str | None
    step: int
    ckpt_key: str
    created_at: str
    bs_cfg: BlackScholesConfig
    cvnn_cfg: CVNNConfig
    sobol_seed: int
    sobol_skip: int
    domain_bounds: dict[str, BoundSpec] | None = None


class GbmTrainer:
    """Reproducible GBM-MC + CVNN trainer."""

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
        _sobol_seed: int | None = None,
        _sobol_skip: int = 0,
    ) -> None:
        base_rng = np.random.default_rng(global_seed)
        torch_seed = int(base_rng.integers(0, 2**63 - 1))
        sobol_seed = _sobol_seed or int(base_rng.integers(0, 2**63 - 1))

        random.seed(global_seed)
        np.random.seed(global_seed)
        torch.manual_seed(torch_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(torch_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.cvnn = cvnn.to(self.device, dtype=_torch_dtype(cfg.sim_params.dtype))

        self.cfg = cfg
        self.domain_bounds = domain_bounds
        self._sobol_seed = sobol_seed
        self._sobol_skip = _sobol_skip

        self._optim: torch.optim.Optimizer | None = None

        if self.device.type == "cuda":
            self.torch_stream: torch.cuda.Stream | None = torch.cuda.Stream(
                device=self.device
            )
        else:
            self.torch_stream = None

        self.sampler = SobolSampler(
            pydantic_class=BlackScholes.Inputs,
            dimensions=self.domain_bounds,
            skip=self._sobol_skip,
            seed=self._sobol_seed,
        )
        self.mc_engine = BlackScholes(cfg)

        self._tb = _TBLogger(tb_logdir, hist_every, flush_every)
        self._s3 = boto3.client("s3", endpoint_url=os.getenv("AWS_ENDPOINT_URL"))
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self._run_prefix = f"{S3_PREFIX}/runs/{ts}"
        self._parent_version: str | None = None

    def train(
        self,
        *,
        num_batches: int,
        batch_size: int,
        learning_rate: float = DEFAULT_LEARNING_RATE,
    ) -> None:
        if self.device.type != "cuda":
            raise RuntimeError("CUDA device required for GBM kernel.")

        opt = torch.optim.Adam(self.cvnn.parameters(), lr=learning_rate)
        self._optim = opt
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

        step = 0
        for _ in range(num_batches):
            t0 = time.perf_counter()

            pts = self.sampler.sample(batch_size)
            self._sobol_skip += batch_size

            fft_buf = cp.zeros(
                (batch_size, self.cfg.sim_params.network_size), dtype=cp_dtype
            )
            for idx, contract in enumerate(pts):
                mc = _price_one(self.mc_engine, contract)
                shaped = mc.put_price.reshape(
                    (
                        self.cfg.sim_params.batches_per_mc_run,
                        self.cfg.sim_params.network_size,
                    )
                )
                # Perform an FFT across axis=1, then average
                fft_buf[idx, :] = cp.mean(cp.fft.fft(shaped, axis=1), axis=0)

            # Wait for CP operations to finish
            cp.cuda.Stream.null.synchronize()

            # Run the forward/backward pass on the PyTorch side
            if self.torch_stream is not None:
                with torch.cuda.stream(self.torch_stream):
                    targets = torch.utils.dlpack.from_dlpack(fft_buf.toDlpack()).to(
                        torch_cdtype
                    )
                    rin, iim = _inputs_to_real_imag(pts, torch_rdtype, self.device)
                    pr, pi = self.cvnn(rin, iim)
                    loss = nn.functional.mse_loss(
                        pr, targets.real
                    ) + nn.functional.mse_loss(pi, targets.imag)

                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    opt.step()

                    grad_norm = nn.utils.clip_grad_norm_(
                        self.cvnn.parameters(), float("inf")
                    ).item()

                self.torch_stream.synchronize()
            else:
                targets = torch.utils.dlpack.from_dlpack(fft_buf.toDlpack()).to(
                    torch_cdtype
                )
                rin, iim = _inputs_to_real_imag(pts, torch_rdtype, self.device)
                pr, pi = self.cvnn(rin, iim)
                loss = nn.functional.mse_loss(
                    pr, targets.real
                ) + nn.functional.mse_loss(pi, targets.imag)

                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()

                grad_norm = nn.utils.clip_grad_norm_(
                    self.cvnn.parameters(), float("inf")
                ).item()

            batch_time = time.perf_counter() - t0
            self._tb.log_step(
                model=self.cvnn,
                step=step,
                loss=float(loss.item()),
                lr=opt.param_groups[0]["lr"],
                grad_norm=grad_norm,
                batch_time=batch_time,
            )
            step += 1

        self._tb.close()
        self._commit_checkpoint(step, opt)

    def predict_price(
        self, inputs: List[BlackScholes.Inputs]
    ) -> List[BlackScholes.HostPricingResults]:
        if not inputs:
            return []
        self.cvnn.eval()

        rdtype = _torch_dtype(self.cfg.sim_params.dtype)
        rin, iim = _inputs_to_real_imag(inputs, rdtype, self.device)

        if self.torch_stream is not None:
            with torch.no_grad(), torch.cuda.stream(self.torch_stream):
                pr, pi = self.cvnn(rin, iim)
            self.torch_stream.synchronize()
        else:
            with torch.no_grad():
                pr, pi = self.cvnn(rin, iim)

        spec = torch.complex(pr, pi)
        mean_ifft = torch.fft.ifft(spec, dim=1).mean(dim=1)

        out: List[BlackScholes.HostPricingResults] = []
        for val, bs in zip(mean_ifft, inputs):
            real_val = float(val.real.item())
            imag_val = float(val.imag.item())
            if abs(imag_val) > IMAG_TOL:
                warnings.warn(f"Imag part {imag_val:.2e} exceeds tol", RuntimeWarning)

            disc = math.exp(-bs.r * bs.T)
            fwd = bs.X0 * math.exp((bs.r - bs.d) * bs.T)

            put_intr = disc * max(bs.K - fwd, 0.0)
            call_val = real_val + fwd - bs.K * disc
            call_intr = disc * max(fwd - bs.K, 0.0)

            out.append(
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
        return out

    def _commit_checkpoint(self, step: int, opt: torch.optim.Optimizer) -> None:
        lock_key = f"{self._run_prefix}/LOCK"
        try:
            self._s3.put_object(
                Bucket=S3_BUCKET, Key=lock_key, Body=b"", IfNoneMatch="*"
            )
        except self._s3.exceptions.ClientError as exc:
            raise RuntimeError("Another writer holds the lock") from exc

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
                bs_cfg=self.mc_engine.snapshot(),
                cvnn_cfg=self.cvnn.as_config(),
                sobol_seed=self._sobol_seed,
                sobol_skip=self._sobol_skip,
                domain_bounds=self.domain_bounds,
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

    @classmethod
    def load_from_uri(
        cls, uri: str, *, device: torch.device | None = None
    ) -> GbmTrainer:
        if uri.startswith("s3://"):
            bucket, key = uri[5:].split("/", 1)
            s3 = boto3.client("s3", endpoint_url=os.getenv("AWS_ENDPOINT_URL"))
            meta_bytes = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
            meta = _Meta.model_validate_json(meta_bytes)
            ckpt_bytes = s3.get_object(Bucket=bucket, Key=meta.ckpt_key)["Body"].read()
        else:
            meta = _Meta.model_validate_json(Path(uri).read_bytes())
            ckpt_bytes = Path(meta.ckpt_key).read_bytes()

        ckpt = torch.load(io.BytesIO(ckpt_bytes), map_location="cpu")
        cvnn = meta.cvnn_cfg.build()
        cvnn.load_state_dict(ckpt["model"])

        domain_bounds = meta.domain_bounds if meta.domain_bounds is not None else {}

        trainer = cls(
            cfg=meta.bs_cfg,
            domain_bounds=domain_bounds,
            cvnn=cvnn,
            device=device,
            global_seed=GLOBAL_SEED_DEFAULT,
            _sobol_seed=meta.sobol_seed,
            _sobol_skip=meta.sobol_skip,
        )
        optim = torch.optim.Adam(trainer.cvnn.parameters())
        optim.load_state_dict(ckpt["optim"])
        trainer._optim = optim
        return trainer
