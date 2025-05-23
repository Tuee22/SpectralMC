# tests/spectralmc/test_gbm_trainer.py
from __future__ import annotations

import io
import json
import os
import uuid
from typing import Dict, Literal, Sequence

import boto3
import numpy as np
import pytest
import torch

from spectralmc.cvnn import CVNN
from spectralmc.gbm import BlackScholes, BlackScholesConfig, SimulationParams
from spectralmc import gbm_trainer as gbt_module
from spectralmc.gbm_trainer import GbmTrainer
from spectralmc.sobol_sampler import BoundSpec

# --------------------------------------------------------------------------- #
# helpers                                                                     #
# --------------------------------------------------------------------------- #


def _bounds() -> Dict[str, BoundSpec]:
    return {
        "X0": BoundSpec(lower=50, upper=150),
        "K": BoundSpec(lower=50, upper=150),
        "T": BoundSpec(lower=0.1, upper=2.0),
        "r": BoundSpec(lower=0.0, upper=0.1),
        "d": BoundSpec(lower=0.0, upper=0.05),
        "v": BoundSpec(lower=0.1, upper=0.5),
    }


def _assert_same_tensors(t1: Sequence[torch.Tensor], t2: Sequence[torch.Tensor]) -> None:
    for a, b in zip(t1, t2, strict=True):
        assert torch.equal(a, b)


# --------------------------------------------------------------------------- #
# main test                                                                   #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("precision", ("float32", "float64"))
def test_real_s3_repro(precision: Literal["float32", "float64"]) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required")

    # ---- connect to the docker-compose MinIO ------------------------------
    s3 = boto3.client(
        "s3",
        endpoint_url=os.getenv("AWS_ENDPOINT_URL", "http://minio:9000"),
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID", "minioadmin"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY", "minioadmin"),
        region_name="us-east-1",
    )

    # ---- create a fresh bucket & monkey-patch the trainer -----------------
    bucket = f"opt-models-test-{uuid.uuid4().hex[:8]}"
    s3.create_bucket(Bucket=bucket)
    original_bucket = gbt_module.S3_BUCKET
    gbt_module.S3_BUCKET = bucket  # type: ignore[assignment]

    try:
        # ------------------------------------------------------------------ #
        # build and train an ORIGINAL trainer                                #
        # ------------------------------------------------------------------ #
        sim_p = SimulationParams(
            timesteps=1,
            network_size=32,
            batches_per_mc_run=2**14,
            threads_per_block=256,
            mc_seed=11,
            buffer_size=1,
            dtype=precision,
        )
        cfg = BlackScholesConfig(sim_params=sim_p)

        net0 = CVNN(6, sim_p.network_size, 64, 1)
        trainer0 = GbmTrainer(
            cfg=cfg,
            domain_bounds=_bounds(),
            cvnn=net0,
            device=torch.device("cuda"),
            global_seed=123,
        )
        trainer0.train(num_batches=8, batch_size=32, learning_rate=5e-3)

        # ------------------------------------------------------------------ #
        # load the checkpoint we just produced                               #
        # ------------------------------------------------------------------ #
        objs = s3.list_objects_v2(Bucket=bucket, Prefix="cvnn/v1/runs/")["Contents"]
        head_key = next(o["Key"] for o in objs if o["Key"].endswith("/HEAD"))
        meta_key = s3.get_object(Bucket=bucket, Key=head_key)["Body"].read().decode()
        meta = json.loads(s3.get_object(Bucket=bucket, Key=meta_key)["Body"].read())
        ckpt_bytes = s3.get_object(Bucket=bucket, Key=meta["ckpt_key"])["Body"].read()
        ckpt = torch.load(io.BytesIO(ckpt_bytes), map_location="cpu")

        # build RESTORED trainer -------------------------------------------
        netR = CVNN(6, sim_p.network_size, 64, 1)
        netR.load_state_dict(ckpt["model"])
        trainerR = GbmTrainer(
            cfg=BlackScholesConfig(**meta["bs_cfg"]),
            domain_bounds=_bounds(),
            cvnn=netR,
            device=torch.device("cuda"),
            global_seed=123,  # same seed path
        )
        optR = torch.optim.Adam(trainerR.cvnn.parameters())
        optR.load_state_dict(ckpt["optim"])
        trainerR._optim = optR  # type: ignore[attr-defined]

        # give trainer0 its optimiser handle too (created inside .train)
        opt0 = torch.optim.Adam(trainer0.cvnn.parameters())
        opt0.load_state_dict(ckpt["optim"])
        trainer0._optim = opt0  # type: ignore[attr-defined]

        # ------------------------------------------------------------------ #
        # identical further training on BOTH instances                       #
        # ------------------------------------------------------------------ #
        for t in (trainer0, trainerR):
            t.train(num_batches=4, batch_size=16, learning_rate=1e-3)

        # CVNN parameters equal --------------------------------------------
        _assert_same_tensors(trainer0.cvnn.state_dict().values(), trainerR.cvnn.state_dict().values())

        # skip counter equal ------------------------------------------------
        skip0 = trainer0.mc_engine.snapshot().sim_params.skip
        skipR = trainerR.mc_engine.snapshot().sim_params.skip
        assert skip0 == skipR

        # quick prediction smoke-test --------------------------------------
        inp = [
            BlackScholes.Inputs(X0=100, K=100, T=1, r=0.02, d=0, v=0.25),
            BlackScholes.Inputs(X0=110, K=90, T=0.5, r=0.03, d=0, v=0.3),
        ]
        out0, outR = trainer0.predict_price(inp), trainerR.predict_price(inp)
        for r0, rR in zip(out0, outR, strict=True):
            assert np.isclose(r0.put_price, rR.put_price)
            assert np.isclose(r0.call_price, rR.call_price)

    finally:
        # teardown bucket unless KEEP_S3_BUCKET=1
        if os.getenv("KEEP_S3_BUCKET") != "1":
            for o in s3.list_objects_v2(Bucket=bucket).get("Contents", []):
                s3.delete_object(Bucket=bucket, Key=o["Key"])
            s3.delete_bucket(Bucket=bucket)
        gbt_module.S3_BUCKET = original_bucket  # restore constant