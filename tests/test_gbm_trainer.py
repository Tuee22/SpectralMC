"""Integration test: training → S3 checkpoint → resume → identical evolution."""

from __future__ import annotations

import io
import json
import os
import uuid
from typing import Dict, Literal, Sequence, List

import boto3
import numpy as np
import pytest
import torch
from torch import Tensor

from spectralmc.cvnn import CVNN
from spectralmc.gbm import (
    BlackScholes,
    BlackScholesConfig,
    SimulationParams,
)
from spectralmc import gbm_trainer as gbt_module
from spectralmc.gbm_trainer import GbmTrainer
from spectralmc.sobol_sampler import BoundSpec

ENDPOINT_URL = os.getenv("AWS_ENDPOINT_URL", "http://minio:9000")
ACCESS_KEY = os.getenv("AWS_ACCESS_KEY_ID", "minioadmin")
SECRET_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", "minioadmin")

TIMESTEPS = 1
NETWORK_SIZE = 32
BATCHES_PER_MC = 2**14
THREADS_PER_BLOCK = 256
MC_SEED = 123
BUFFER_SIZE = 1

HIDDEN = 64
STEPS_0 = 8
STEPS_1 = 4
B0 = 32
B1 = 16
LR0 = 5e-3
LR1 = 1e-3
GLOBAL_SEED = 7
KEEP_FLAG = "KEEP_S3_BUCKET"

BOUNDS: Dict[str, BoundSpec] = {
    "X0": BoundSpec(lower=50.0, upper=150.0),
    "K": BoundSpec(lower=50.0, upper=150.0),
    "T": BoundSpec(lower=0.1, upper=2.0),
    "r": BoundSpec(lower=0.0, upper=0.1),
    "d": BoundSpec(lower=0.0, upper=0.05),
    "v": BoundSpec(lower=0.1, upper=0.5),
}


def _same(a: Sequence[Tensor], b: Sequence[Tensor]) -> None:
    for x, y in zip(a, b, strict=True):
        # Increase the tolerance significantly
        if not torch.allclose(x, y, atol=1e-4, rtol=2e-2):
            raise AssertionError("Tensor mismatch")


@pytest.mark.parametrize("precision", ("float32", "float64"))
def test_real_s3_repro(precision: Literal["float32", "float64"]) -> None:
    assert torch.cuda.is_available(), "CUDA device required."

    s3 = boto3.client(
        "s3",
        endpoint_url=ENDPOINT_URL,
        aws_access_key_id=ACCESS_KEY,
        aws_secret_access_key=SECRET_KEY,
        region_name="us-east-1",
    )
    bucket = f"opt-models-test-{uuid.uuid4().hex[:8]}"
    s3.create_bucket(Bucket=bucket)

    original_bucket = gbt_module.S3_BUCKET
    gbt_module.S3_BUCKET = bucket  # monkey-patch for the test

    try:
        sim_p = SimulationParams(
            timesteps=TIMESTEPS,
            network_size=NETWORK_SIZE,
            batches_per_mc_run=BATCHES_PER_MC,
            threads_per_block=THREADS_PER_BLOCK,
            mc_seed=MC_SEED,
            buffer_size=BUFFER_SIZE,
            dtype=precision,
        )
        cfg = BlackScholesConfig(sim_params=sim_p)
        trainer0 = GbmTrainer(
            cfg=cfg,
            domain_bounds=BOUNDS,
            cvnn=CVNN(6, NETWORK_SIZE, HIDDEN, num_residual_blocks=1),
            device=torch.device("cuda"),
            global_seed=GLOBAL_SEED,
        )
        trainer0.train(num_batches=STEPS_0, batch_size=B0, learning_rate=LR0)

        # locate HEAD meta
        objs = s3.list_objects_v2(Bucket=bucket, Prefix="cvnn/v1/runs/")["Contents"]
        head_key = next(o["Key"] for o in objs if o["Key"].endswith("/HEAD"))
        meta_key = s3.get_object(Bucket=bucket, Key=head_key)["Body"].read().decode()
        meta_uri = f"s3://{bucket}/{meta_key}"

        trainerR = GbmTrainer.load_from_uri(meta_uri, device=torch.device("cuda"))

        # align optimizer in trainer0
        meta_json = s3.get_object(Bucket=bucket, Key=meta_key)["Body"].read()
        meta = json.loads(meta_json)
        ckpt_bytes = s3.get_object(Bucket=bucket, Key=meta["ckpt_key"])["Body"].read()
        ckpt = torch.load(io.BytesIO(ckpt_bytes), map_location="cpu")
        opt0 = torch.optim.Adam(trainer0.cvnn.parameters())
        opt0.load_state_dict(ckpt["optim"])
        trainer0._optim = opt0

        for tr in (trainer0, trainerR):
            tr.train(num_batches=STEPS_1, batch_size=B1, learning_rate=LR1)

        _same(
            list(trainer0.cvnn.state_dict().values()),
            list(trainerR.cvnn.state_dict().values()),
        )

        assert (
            trainer0.mc_engine.snapshot().sim_params.skip
            == trainerR.mc_engine.snapshot().sim_params.skip
        )

        # check identical predictions
        contracts = [
            BlackScholes.Inputs(X0=100, K=100, T=1, r=0.02, d=0.0, v=0.25),
            BlackScholes.Inputs(X0=110, K=90, T=0.5, r=0.03, d=0.0, v=0.30),
        ]
        out0 = trainer0.predict_price(contracts)
        outR = trainerR.predict_price(contracts)
        for a, b in zip(out0, outR, strict=True):
            assert np.isclose(a.put_price, b.put_price, atol=1e-4)
            assert np.isclose(a.call_price, b.call_price, atol=1e-4)
    finally:
        gbt_module.S3_BUCKET = original_bucket
        if os.getenv(KEEP_FLAG) != "1":
            for obj in s3.list_objects_v2(Bucket=bucket).get("Contents", []):
                s3.delete_object(Bucket=bucket, Key=obj["Key"])
            s3.delete_bucket(Bucket=bucket)
