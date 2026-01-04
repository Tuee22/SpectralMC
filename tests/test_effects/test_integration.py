"""Integration tests for SpectralMCInterpreter and effect registry."""

from __future__ import annotations

import pytest
import torch
import cupy as cp
from torch import nn

from spectralmc.effects import (
    BackwardPass,
    ComputeComplexLoss,
    ComputeGradNorm,
    ForwardPassComplex,
    GenerateNormals,
    LogMetrics,
    OptimizerStep,
    SampleContracts,
    SpectralMCInterpreter,
    SplitInputs,
    StreamSync,
    ZeroGrad,
    sequence_effects,
)
from spectralmc.effects.registry import SharedRegistry
from spectralmc.result import Failure, Success
from spectralmc.sobol_sampler import SobolSampler, build_sobol_config
from tests.helpers import make_domain_bounds
from spectralmc.gbm import BlackScholes


assert torch.cuda.is_available()


class _ToyComplexModel(nn.Module):
    """Minimal complex-valued model for testing."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(3, 1, bias=False)

    def forward(self, real: torch.Tensor, imag: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        real_out = self.linear(real)
        imag_out = self.linear(imag)
        return real_out, imag_out


@pytest.mark.asyncio
async def test_interpreter_creation() -> None:
    interpreter = SpectralMCInterpreter.create(
        torch_stream=torch.cuda.Stream(),
        cupy_stream=cp.cuda.Stream(),
        storage_bucket="test-bucket",
    )

    assert interpreter.gpu_interpreter is not None
    assert interpreter.training_interpreter is not None
    assert interpreter.montecarlo_interpreter is not None
    assert interpreter.storage_interpreter is not None
    assert interpreter.rng_interpreter is not None
    assert interpreter.metadata_interpreter is not None
    assert interpreter.logging_interpreter is not None
    assert interpreter.registry is not None


def test_registry_tensor_flow() -> None:
    registry = SharedRegistry()

    tensor = torch.randn(2, 3, device="cuda")
    assert isinstance(registry.register_tensor("t", tensor), Success)

    retrieved = registry.get_tensor("t")
    assert isinstance(retrieved, Success)
    assert isinstance(retrieved.value, torch.Tensor)
    assert torch.equal(retrieved.value, tensor)


def test_registry_model_flow() -> None:
    registry = SharedRegistry()
    model = _ToyComplexModel().cuda()

    assert isinstance(registry.register_model("model", model), Success)

    retrieved = registry.get_model("model")
    assert isinstance(retrieved, Success)
    assert retrieved.value is model


@pytest.mark.asyncio
async def test_interpret_sequence_success() -> None:
    interpreter = SpectralMCInterpreter.create(
        torch_stream=torch.cuda.Stream(),
        cupy_stream=cp.cuda.Stream(),
        storage_bucket="",
    )

    effects = sequence_effects(
        GenerateNormals(rows=2, cols=2, seed=7, output_tensor_id="normals"),
        StreamSync(stream_type="cupy"),
    )

    result = await interpreter.interpret_sequence(effects)
    assert isinstance(result, Success)
    normals = interpreter.registry.get_tensor("normals")
    assert isinstance(normals, Success)
    assert isinstance(normals.value, (cp.ndarray, torch.Tensor))
    assert normals.value.shape == (2, 2)


@pytest.mark.asyncio
async def test_interpret_sequence_failure_propagation() -> None:
    interpreter = SpectralMCInterpreter.create(
        torch_stream=torch.cuda.Stream(),
        cupy_stream=cp.cuda.Stream(),
        storage_bucket="",
    )

    effects = sequence_effects(
        StreamSync(stream_type="torch"),
        OptimizerStep(optimizer_id="missing"),
    )

    result = await interpreter.interpret_sequence(effects)
    assert isinstance(result, Failure)


@pytest.mark.asyncio
async def test_training_effect_sequence() -> None:
    interpreter = SpectralMCInterpreter.create(
        torch_stream=torch.cuda.Stream(),
        cupy_stream=cp.cuda.Stream(),
        storage_bucket="",
    )

    model = _ToyComplexModel().cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    registry = interpreter.registry
    assert isinstance(registry.register_model("cvnn", model), Success)
    assert isinstance(registry.register_optimizer("adam", optimizer), Success)

    real_input = torch.randn(4, 3, device="cuda")
    imag_input = torch.randn(4, 3, device="cuda")
    target = torch.randn(4, 1, device="cuda")
    registry.register_tensor("real_in", real_input)
    registry.register_tensor("imag_in", imag_input)
    registry.register_tensor("targets", target)

    effects = sequence_effects(
        ZeroGrad(optimizer_id="adam"),
        ForwardPassComplex(
            model_id="cvnn",
            real_input_tensor_id="real_in",
            imag_input_tensor_id="imag_in",
            real_output_tensor_id="pred_real",
            imag_output_tensor_id="pred_imag",
        ),
        ComputeComplexLoss(
            pred_real_tensor_id="pred_real",
            pred_imag_tensor_id="pred_imag",
            target_tensor_id="targets",
            loss_type="mse",
            output_tensor_id="loss",
        ),
        BackwardPass(loss_tensor_id="loss"),
        ComputeGradNorm(model_id="cvnn", output_tensor_id="grad_norm"),
        OptimizerStep(optimizer_id="adam"),
        StreamSync(stream_type="torch"),
        LogMetrics(metrics=(("loss", "{loss}"),), step=0),
    )

    result = await interpreter.interpret_sequence(effects)
    assert isinstance(result, Success)
    loss = registry.get_tensor("loss")
    grad_norm = registry.get_tensor("grad_norm")

    assert isinstance(loss, Success)
    assert isinstance(grad_norm, Success)
    assert isinstance(loss.value, torch.Tensor)
    assert isinstance(grad_norm.value, torch.Tensor)
    assert loss.value.ndim == 0
    assert grad_norm.value.ndim == 0


@pytest.mark.asyncio
async def test_sample_contracts_and_split_inputs() -> None:
    sampler_config = build_sobol_config(seed=1, skip=0).unwrap()
    sampler = SobolSampler.create(
        pydantic_class=BlackScholes.Inputs,
        dimensions=make_domain_bounds(),
        config=sampler_config,
    ).unwrap()

    interpreter = SpectralMCInterpreter.create(
        torch_stream=torch.cuda.Stream(),
        cupy_stream=cp.cuda.Stream(),
        storage_bucket="",
    )

    interpreter.registry.register_sampler("sobol_sampler", sampler)

    effects = sequence_effects(
        SampleContracts(sampler_id="sobol_sampler", num_samples=2, output_tensor_id="contracts"),
        SplitInputs(
            contracts_tensor_id="contracts",
            real_output_tensor_id="real_inputs",
            imag_output_tensor_id="imag_inputs",
        ),
    )

    result = await interpreter.interpret_sequence(effects)
    assert isinstance(result, Success)

    real_inputs = interpreter.registry.get_tensor("real_inputs")
    imag_inputs = interpreter.registry.get_tensor("imag_inputs")
    assert isinstance(real_inputs, Success)
    assert isinstance(imag_inputs, Success)
    assert isinstance(real_inputs.value, torch.Tensor)
    assert isinstance(imag_inputs.value, torch.Tensor)
    assert real_inputs.value.shape[0] == 2
    assert torch.allclose(imag_inputs.value, torch.zeros_like(imag_inputs.value))
