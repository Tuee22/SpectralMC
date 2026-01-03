# Effect Interpreter Migration Plan - Phase 4 Implementation

**Status**: ACTIVE
**Created**: 2026-01-03
**Target Completion**: ~5-6 weeks
**Related Documentation**:
- `documents/engineering/effect_interpreter.md` - Effect Interpreter doctrine
- `documents/engineering/purity_doctrine.md` - Tier classification system
- `documents/engineering/coding_standards.md` - ADT patterns and Result types

---

## Executive Summary

This document outlines the complete migration path from **Phase 2 (Effect Builders + Direct Execution)** to **Phase 4 (Full Effect System)** as defined in `documents/engineering/effect_interpreter.md:1242-1249`.

### Current Status
- **Compliance**: ✅ FULLY COMPLIANT with tier classification
  - `gbm_trainer.py` is Tier 3 (Effect Interpreter) - explicitly exempt from strict purity
  - Direct GPU operations are ALLOWED in Tier 3
  - Current imperative `train()` implementation is ACCEPTABLE

### Documentation Goal
- **Target**: **Phase 4 - Full Effect System** (`effect_interpreter.md:1242-1249`)
  - "All side effects modeled as ADTs"
  - "Single interpreter entry point"
  - "Pure code separated from effectful execution"
  - "Deprecate direct side-effect APIs"

### Gap Analysis
- **Current**: Phase 2 - Effect builders exist but `train()` uses direct GPU operations
- **Target**: Phase 4 - All training logic executed through `SpectralMCInterpreter`
- **Migration Required**: 20 specific changes across 12 files

---

## Table of Contents

1. [What Must Change: Summary](#what-must-change-summary)
2. [Detailed Implementation Plan](#detailed-implementation-plan)
   - [Step 1: Foundation (2-3 days)](#step-1-foundation-2-3-days)
   - [Step 2: Architecture (1-2 weeks)](#step-2-architecture-1-2-weeks)
   - [Step 3: Effect Builder Rewrite (1 week)](#step-3-effect-builder-rewrite-1-week)
   - [Step 4: Training Migration (1-2 weeks)](#step-4-training-migration-1-2-weeks)
   - [Step 5: Logging Migration (1-2 days)](#step-5-logging-migration-1-2-days)
3. [Timeline and Milestones](#timeline-and-milestones)
4. [Success Criteria](#success-criteria)
5. [Risk Mitigation](#risk-mitigation)
6. [Files to Modify](#files-to-modify)
7. [Technical Deep Dives](#technical-deep-dives)

---

## What Must Change: Summary

Based on thorough review of `documents/engineering/`, here are the 20 specific changes required to reach **Phase 4 (Full Effect System)**:

### 1. Effect System Foundation (3 changes)

**Current Problem**: Effect system is 90% theoretical - `SpectralMCInterpreter` has ZERO integration tests.

**Required Changes**:
1. ✅ Add integration tests for `SpectralMCInterpreter` (currently ZERO tests)
2. ✅ Test `SharedRegistry` data flow (currently untested)
3. ✅ Fix `TrainingInterpreter._optimizer_step()` bug (calls `zero_grad()` in wrong order)

**Why Critical**: Must prove effect system works before building migration on top of it.

---

### 2. Missing Effect Types (11 new effects)

**Current Problem**: Effect system cannot express training workflow - missing 11 critical effect types.

**Required New Effects**:

#### Training Effects (4 new types)
1. ✅ `ForwardPassComplex` - Dual real/imag inputs for CVNN (current `ForwardPass` only handles single tensor)
2. ✅ `ComputeComplexLoss` - Dual MSE computation for real/imag outputs
3. ✅ `ZeroGrad` - Separate from `OptimizerStep` (current bug: wrong order)
4. ✅ `ComputeGradNorm` - Gradient monitoring for logging

#### Monte Carlo Effects (4 new types)
5. ✅ `SampleContracts` - Sobol sampling integration (currently hardcoded params)
6. ✅ `StackTensors` - Batch result aggregation
7. ✅ `ComputeMeanFFT` - FFT with mean reduction
8. ✅ `ProcessBatch` - High-level composite effect for per-contract processing

#### GPU Effects (1 new type)
9. ✅ `SplitInputs` - Contract → real/imag tensors

#### Storage Effects (1 new type)
10. ✅ `CommitCheckpoint` - Conditional blockchain commits based on `CommitPlan`

#### Optional (1 new type)
11. ✅ `UpdateLearningRate` - LR scheduler (if needed in future)

**Why Critical**: Cannot model training workflow without these effects. Effect system currently cannot express:
- Complex-valued neural network forward pass (dual inputs/outputs)
- Per-contract Monte Carlo processing (loops not expressible in pure ADTs)
- Conditional blockchain commits

---

### 3. Effect Builder Fixes (1 major rewrite)

**Current Problem**: `build_training_step_effects()` has 11 mismatches with actual training workflow.

**Specific Issues**:
- Uses non-existent tensor IDs (`batch_{idx}` vs actual IDs)
- Hardcoded market parameters (S=100, r=0.05) instead of Sobol-sampled
- Missing per-contract processing logic (loop over contracts)
- Incorrect effect sequence (doesn't match `_run_batch()` implementation)
- Missing gradient norm computation
- Missing metadata updates for Sobol skip tracking

**Required Change**:
- ✅ Rewrite `build_training_step_effects()` to match `_run_batch()` line-by-line

**File**: `src/spectralmc/gbm_trainer.py:907-1024`

**Why Critical**: Effect builder is the blueprint for training. If it doesn't match reality, interpreter will execute wrong workflow.

---

### 4. Training Loop Migration (6 changes)

**Current Problem**: `train()` uses imperative code with direct GPU operations (`.backward()`, `.step()`, CUDA kernel launches).

**Required Changes**:
1. ✅ Implement `train_via_effects()` using `interpreter.interpret_sequence()`
2. ✅ Migrate `train()` to delegate to `train_via_effects()`
3. ✅ Delete `_run_batch()` (lines 1533-1600) - direct GPU operations
4. ✅ Delete `_torch_step()` (lines 820-836) - direct `.backward()`, `.step()` calls
5. ✅ Delete `_simulate_fft()` direct GPU usage (lines 807-819)
6. ✅ Remove all direct GPU operations from training orchestration

**Why Critical**: This is the core migration - moving from imperative to declarative training execution. High risk (touching working code with 100% test coverage).

---

### 5. Logging Migration (2 changes)

**Current Problem**: 28 direct `logger.*()` calls in Tier 3 storage code violate Phase 4 requirement.

**Required Changes**:
1. ✅ Training logging already uses `LogMetrics` effects (just needs correct implementation from Step 3)
2. ✅ Convert 28 logger calls in storage layer:
   - `src/spectralmc/storage/tensorboard_writer.py` - 10 calls
   - `src/spectralmc/storage/inference.py` - 15 calls
   - `src/spectralmc/storage/store.py` - 2 calls
   - `src/spectralmc/storage/gc.py` - 1 call

**Why Critical**: Phase 4 requires "All side effects modeled as ADTs" - logging is a side effect.

---

**Total**: 20 specific changes required across 5 major categories

---

## Detailed Implementation Plan

### Step 1: Foundation (2-3 days)

**Goal**: Prove effect system works before building on it

**Why This Step Matters**:
- Effect system is currently 90% theoretical
- `SpectralMCInterpreter` has ZERO integration tests
- Must validate architecture before investing weeks in migration

#### Task 1.1: Create Integration Test Suite

**File**: `tests/test_effects/test_integration.py` (NEW FILE)

**Test Coverage Required**:

```python
"""Integration tests for SpectralMCInterpreter and effect system."""

import pytest
import torch
from spectralmc.effects import (
    SpectralMCInterpreter,
    SharedRegistry,
    GenerateNormals,
    TensorTransfer,
    StreamSync,
    ForwardPass,
    ComputeLoss,
    BackwardPass,
    OptimizerStep,
    sequence_effects,
)
from spectralmc.result import Success, Failure


@pytest.mark.asyncio
async def test_interpreter_creation():
    """Test SpectralMCInterpreter.create() initializes all sub-interpreters."""
    interpreter = SpectralMCInterpreter.create(
        torch_stream=torch.cuda.current_stream(),
        cupy_stream=None,
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


@pytest.mark.asyncio
async def test_registry_tensor_flow():
    """Test SharedRegistry stores and retrieves tensors correctly."""
    registry = SharedRegistry()

    # Register tensor
    tensor = torch.randn(10, 5, device="cuda")
    registry.register_tensor("test_tensor", tensor)

    # Retrieve tensor
    retrieved = registry.get_tensor("test_tensor")
    assert torch.equal(retrieved, tensor)
    assert retrieved.device == tensor.device


@pytest.mark.asyncio
async def test_registry_model_flow():
    """Test SharedRegistry stores and retrieves models correctly."""
    registry = SharedRegistry()

    # Register model
    model = torch.nn.Linear(10, 5).cuda()
    registry.register_model("test_model", model)

    # Retrieve model
    retrieved = registry.get_model("test_model")
    assert retrieved is model
    assert next(retrieved.parameters()).device.type == "cuda"


@pytest.mark.asyncio
async def test_interpret_sequence_success():
    """Test interpret_sequence executes multiple effects sequentially."""
    interpreter = SpectralMCInterpreter.create(
        torch_stream=torch.cuda.current_stream(),
        cupy_stream=None,
        storage_bucket="test-bucket",
    )

    # Build effect sequence
    effects = sequence_effects(
        GenerateNormals(rows=100, cols=10, seed=42, output_tensor_id="normals"),
        TensorTransfer(
            source_tensor_id="normals",
            target_device="cuda",
            output_tensor_id="normals_gpu",
        ),
        StreamSync(stream_type="torch"),
    )

    # Execute
    result = await interpreter.interpret_sequence(effects)

    # Validate
    assert isinstance(result, Success)
    normals = interpreter.registry.get_tensor("normals_gpu")
    assert normals.shape == (100, 10)
    assert normals.device.type == "cuda"


@pytest.mark.asyncio
async def test_interpret_sequence_failure_propagation():
    """Test interpret_sequence propagates failures correctly."""
    interpreter = SpectralMCInterpreter.create(
        torch_stream=torch.cuda.current_stream(),
        cupy_stream=None,
        storage_bucket="test-bucket",
    )

    # Build effect sequence with invalid tensor ID
    effects = sequence_effects(
        TensorTransfer(
            source_tensor_id="nonexistent_tensor",
            target_device="cuda",
            output_tensor_id="output",
        ),
    )

    # Execute
    result = await interpreter.interpret_sequence(effects)

    # Validate failure
    assert isinstance(result, Failure)
    assert "nonexistent_tensor" in str(result.error)


@pytest.mark.asyncio
async def test_training_effect_sequence():
    """Test complete training step via effects."""
    interpreter = SpectralMCInterpreter.create(
        torch_stream=torch.cuda.current_stream(),
        cupy_stream=None,
        storage_bucket="test-bucket",
    )

    # Setup
    model = torch.nn.Linear(10, 1).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    interpreter.registry.register_model("model", model)
    interpreter.registry.register_optimizer("adam", optimizer)

    # Register input/target tensors
    x = torch.randn(32, 10, device="cuda")
    y = torch.randn(32, 1, device="cuda")
    interpreter.registry.register_tensor("input", x)
    interpreter.registry.register_tensor("target", y)

    # Build training step
    effects = sequence_effects(
        ForwardPass(
            model_id="model",
            input_tensor_id="input",
            output_tensor_id="output",
        ),
        ComputeLoss(
            pred_tensor_id="output",
            target_tensor_id="target",
            loss_type="mse",
            output_tensor_id="loss",
        ),
        BackwardPass(loss_tensor_id="loss"),
        OptimizerStep(optimizer_id="adam"),
        StreamSync(stream_type="torch"),
    )

    # Execute
    result = await interpreter.interpret_sequence(effects)

    # Validate
    assert isinstance(result, Success)
    loss = interpreter.registry.get_tensor("loss")
    assert loss.ndim == 0  # Scalar loss
    assert loss.device.type == "cuda"
```

**Expected Test Count**: 10-15 integration tests covering:
- Interpreter creation and initialization
- Registry operations (tensors, models, optimizers)
- Effect sequence execution
- Success/Failure propagation
- Each sub-interpreter with real effects

#### Task 1.2: Fix Interpreter Bug

**File**: `src/spectralmc/effects/interpreter.py:442-460`

**Current Bug**:
```python
async def _optimizer_step(self, optimizer_id: str):
    optimizer = self._registry.get_optimizer(optimizer_id)
    if hasattr(optimizer, "step"):
        step_fn = optimizer.step
        step_fn()  # Execute step
    if hasattr(optimizer, "zero_grad"):
        zero_grad_fn = optimizer.zero_grad
        zero_grad_fn()  # BUG: zero_grad AFTER step is wrong order!
    return Success(None)
```

**Fix**:
```python
async def _optimizer_step(self, optimizer_id: str):
    """Execute optimizer step (gradients must be computed first via BackwardPass).

    Note: zero_grad() should be called BEFORE forward pass, not here.
    This is now handled by the separate ZeroGrad effect.
    """
    optimizer = self._registry.get_optimizer(optimizer_id)
    if hasattr(optimizer, "step"):
        step_fn = optimizer.step
        step_fn()
    # DO NOT call zero_grad here - use ZeroGrad effect before forward pass
    return Success(None)
```

**Why**: PyTorch training loop order is:
1. `optimizer.zero_grad()` - BEFORE forward pass
2. Forward pass
3. Loss computation
4. `loss.backward()` - Compute gradients
5. `optimizer.step()` - Update parameters

Current code calls `zero_grad()` AFTER `step()`, which clears gradients for next iteration but violates effect semantics (each effect should do one thing).

**Validation**: Add unit test verifying `OptimizerStep` effect doesn't call `zero_grad()`.

#### Deliverables

- [ ] `tests/test_effects/test_integration.py` with 10-15 passing tests
- [ ] `TrainingInterpreter._optimizer_step()` bug fixed
- [ ] All existing tests still pass (361+ tests)
- [ ] Effect system proven to work with real GPU operations

**Exit Criteria**: Can execute multi-effect sequences through `SpectralMCInterpreter` and get correct results.

---

### Step 2: Architecture (1-2 weeks)

**Goal**: Add missing effect types and interpreter methods

**Why This Step Matters**:
- Current effect system cannot express training workflow
- Missing 11 critical effect types for CVNN training
- Cannot proceed with migration without these effects

#### Task 2.1: Add Training Effects

**File**: `src/spectralmc/effects/training.py`

**New Effect Types**:

```python
@dataclass(frozen=True)
class ForwardPassComplex:
    """Request to execute forward pass with complex-valued inputs (real + imaginary).

    Complex-valued neural networks (CVNN) require separate real and imaginary inputs,
    producing separate real and imaginary outputs. This effect models the dual-tensor
    forward pass required for CVNN training.

    Attributes:
        kind: Discriminator for pattern matching. Always "ForwardPassComplex".
        model_id: Identifier for the CVNN model.
        real_input_tensor_id: Identifier for the real component input tensor.
        imag_input_tensor_id: Identifier for the imaginary component input tensor.
        real_output_tensor_id: Identifier for storing the real component output.
        imag_output_tensor_id: Identifier for storing the imaginary component output.
    """
    kind: Literal["ForwardPassComplex"] = "ForwardPassComplex"
    model_id: str = ""
    real_input_tensor_id: str = ""
    imag_input_tensor_id: str = ""
    real_output_tensor_id: str = "pred_real"
    imag_output_tensor_id: str = "pred_imag"


@dataclass(frozen=True)
class ComputeComplexLoss:
    """Request to compute loss for complex-valued predictions.

    CVNN training requires computing loss on both real and imaginary components.
    This effect computes MSE loss as: loss = MSE(pred_real, target) + MSE(pred_imag, 0).

    Attributes:
        kind: Discriminator for pattern matching. Always "ComputeComplexLoss".
        pred_real_tensor_id: Identifier for the real component prediction.
        pred_imag_tensor_id: Identifier for the imaginary component prediction.
        target_tensor_id: Identifier for the target tensor (real-valued).
        loss_type: Type of loss function ("mse", "mae", "huber").
        output_tensor_id: Identifier for storing the computed loss.
    """
    kind: Literal["ComputeComplexLoss"] = "ComputeComplexLoss"
    pred_real_tensor_id: str = ""
    pred_imag_tensor_id: str = ""
    target_tensor_id: str = ""
    loss_type: Literal["mse", "mae", "huber"] = "mse"
    output_tensor_id: str = "loss"


@dataclass(frozen=True)
class ZeroGrad:
    """Request to zero out gradients in optimizer.

    This effect explicitly models the optimizer.zero_grad() call that must happen
    BEFORE the forward pass. Separating this from OptimizerStep fixes the ordering bug
    and makes the effect sequence match PyTorch training loop semantics.

    Attributes:
        kind: Discriminator for pattern matching. Always "ZeroGrad".
        optimizer_id: Identifier for the optimizer.
        set_to_none: If True, set gradients to None instead of zero (more efficient).
    """
    kind: Literal["ZeroGrad"] = "ZeroGrad"
    optimizer_id: str = ""
    set_to_none: bool = True


@dataclass(frozen=True)
class ComputeGradNorm:
    """Request to compute gradient norm for monitoring.

    Gradient norm is critical for debugging training stability. This effect computes
    the L2 norm of all gradients in a model, which is logged for monitoring.

    Attributes:
        kind: Discriminator for pattern matching. Always "ComputeGradNorm".
        model_id: Identifier for the model.
        max_norm: Maximum norm for clipping (default: no clipping).
        output_tensor_id: Identifier for storing the gradient norm scalar.
    """
    kind: Literal["ComputeGradNorm"] = "ComputeGradNorm"
    model_id: str = ""
    max_norm: float = float("inf")
    output_tensor_id: str = "grad_norm"


# Update TrainingEffect union
TrainingEffect = (
    ForwardPass
    | BackwardPass
    | OptimizerStep
    | ComputeLoss
    | LogMetrics
    | ForwardPassComplex  # NEW
    | ComputeComplexLoss  # NEW
    | ZeroGrad  # NEW
    | ComputeGradNorm  # NEW
)
```

**Interpreter Implementation** (`src/spectralmc/effects/interpreter.py`):

```python
class TrainingInterpreter:
    async def _forward_pass_complex(
        self,
        model_id: str,
        real_input_tensor_id: str,
        imag_input_tensor_id: str,
        real_output_tensor_id: str,
        imag_output_tensor_id: str,
    ) -> Result[None, TrainingError]:
        """Execute complex-valued forward pass."""
        model = self._registry.get_model(model_id)
        real_input = self._registry.get_tensor(real_input_tensor_id)
        imag_input = self._registry.get_tensor(imag_input_tensor_id)

        # CVNN forward pass expects (real, imag) tuple
        pred_real, pred_imag = model(real_input, imag_input)

        # Register outputs
        self._registry.register_tensor(real_output_tensor_id, pred_real)
        self._registry.register_tensor(imag_output_tensor_id, pred_imag)

        return Success(None)

    async def _compute_complex_loss(
        self,
        pred_real_tensor_id: str,
        pred_imag_tensor_id: str,
        target_tensor_id: str,
        loss_type: str,
        output_tensor_id: str,
    ) -> Result[None, TrainingError]:
        """Compute loss for complex-valued predictions."""
        pred_real = self._registry.get_tensor(pred_real_tensor_id)
        pred_imag = self._registry.get_tensor(pred_imag_tensor_id)
        target = self._registry.get_tensor(target_tensor_id)

        # Compute dual MSE loss
        if loss_type == "mse":
            loss_real = torch.nn.functional.mse_loss(pred_real, target)
            loss_imag = torch.nn.functional.mse_loss(pred_imag, torch.zeros_like(pred_imag))
            loss = loss_real + loss_imag
        elif loss_type == "mae":
            loss_real = torch.nn.functional.l1_loss(pred_real, target)
            loss_imag = torch.nn.functional.l1_loss(pred_imag, torch.zeros_like(pred_imag))
            loss = loss_real + loss_imag
        else:
            return Failure(TrainingError(message=f"Unknown loss type: {loss_type}"))

        self._registry.register_tensor(output_tensor_id, loss)
        return Success(None)

    async def _zero_grad(
        self,
        optimizer_id: str,
        set_to_none: bool,
    ) -> Result[None, TrainingError]:
        """Zero out gradients."""
        optimizer = self._registry.get_optimizer(optimizer_id)
        optimizer.zero_grad(set_to_none=set_to_none)
        return Success(None)

    async def _compute_grad_norm(
        self,
        model_id: str,
        max_norm: float,
        output_tensor_id: str,
    ) -> Result[None, TrainingError]:
        """Compute gradient norm."""
        model = self._registry.get_model(model_id)

        # Compute total gradient norm
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5

        # Store as tensor
        grad_norm_tensor = torch.tensor(total_norm, device="cuda")
        self._registry.register_tensor(output_tensor_id, grad_norm_tensor)

        return Success(None)

    async def interpret(self, effect: TrainingEffect) -> Result[None, TrainingError]:
        """Interpret training effect (updated with new effects)."""
        match effect.kind:
            case "ForwardPass":
                return await self._forward_pass(...)
            case "ForwardPassComplex":  # NEW
                return await self._forward_pass_complex(
                    effect.model_id,
                    effect.real_input_tensor_id,
                    effect.imag_input_tensor_id,
                    effect.real_output_tensor_id,
                    effect.imag_output_tensor_id,
                )
            case "BackwardPass":
                return await self._backward_pass(...)
            case "OptimizerStep":
                return await self._optimizer_step(...)
            case "ComputeLoss":
                return await self._compute_loss(...)
            case "ComputeComplexLoss":  # NEW
                return await self._compute_complex_loss(
                    effect.pred_real_tensor_id,
                    effect.pred_imag_tensor_id,
                    effect.target_tensor_id,
                    effect.loss_type,
                    effect.output_tensor_id,
                )
            case "ZeroGrad":  # NEW
                return await self._zero_grad(effect.optimizer_id, effect.set_to_none)
            case "ComputeGradNorm":  # NEW
                return await self._compute_grad_norm(
                    effect.model_id,
                    effect.max_norm,
                    effect.output_tensor_id,
                )
            case "LogMetrics":
                return await self._log_metrics(...)
            case _:
                assert_never(effect)
```

**Tests**: Unit test each new effect with `MockInterpreter` and real `TrainingInterpreter`.

#### Task 2.2: Add Monte Carlo Effects

**File**: `src/spectralmc/effects/montecarlo.py`

**New Effect Types**:

```python
@dataclass(frozen=True)
class SampleContracts:
    """Request to sample contracts from Sobol sampler.

    SpectralMC training samples contract parameters (S, K, T, r, σ) from a Sobol
    quasi-random sequence. This effect models the sampling operation.

    Attributes:
        kind: Discriminator for pattern matching. Always "SampleContracts".
        sampler_id: Identifier for the Sobol sampler.
        num_samples: Number of contracts to sample.
        output_tensor_id: Identifier for storing the sampled contracts tensor.
    """
    kind: Literal["SampleContracts"] = "SampleContracts"
    sampler_id: str = ""
    num_samples: int = 0
    output_tensor_id: str = "contracts"


@dataclass(frozen=True)
class ProcessBatch:
    """Request to process a batch of contracts (composite effect).

    Per-contract processing involves:
    1. For each contract (S, K, T, r, σ):
       a. Generate normals
       b. Simulate GBM paths
       c. Compute FFT
       d. Stack results

    This is a high-level composite effect that the interpreter expands into
    per-contract effect sequences. This is necessary because effect ADTs cannot
    express loops directly (purity constraint).

    Attributes:
        kind: Discriminator for pattern matching. Always "ProcessBatch".
        contracts_tensor_id: Identifier for the contracts tensor.
        batch_idx: Batch index for unique tensor IDs.
        config: Training configuration (num_paths, num_steps, etc.).
    """
    kind: Literal["ProcessBatch"] = "ProcessBatch"
    contracts_tensor_id: str = ""
    batch_idx: int = 0
    config: dict = field(default_factory=dict)  # TrainingConfig as dict


@dataclass(frozen=True)
class StackTensors:
    """Request to stack multiple tensors along a dimension.

    Batch processing produces multiple FFT result tensors that must be stacked
    into a single batch tensor for training.

    Attributes:
        kind: Discriminator for pattern matching. Always "StackTensors".
        input_tensor_ids: Tuple of tensor IDs to stack.
        dim: Dimension along which to stack.
        output_tensor_id: Identifier for storing the stacked tensor.
    """
    kind: Literal["StackTensors"] = "StackTensors"
    input_tensor_ids: tuple[str, ...] = ()
    dim: int = 0
    output_tensor_id: str = "stacked"


@dataclass(frozen=True)
class ComputeMeanFFT:
    """Request to compute FFT with mean reduction.

    Attributes:
        kind: Discriminator for pattern matching. Always "ComputeMeanFFT".
        input_tensor_id: Identifier for the input tensor.
        output_tensor_id: Identifier for storing the FFT result.
    """
    kind: Literal["ComputeMeanFFT"] = "ComputeMeanFFT"
    input_tensor_id: str = ""
    output_tensor_id: str = "fft_result"


# Update MonteCarloEffect union
MonteCarloEffect = (
    GenerateNormals
    | SimulatePaths
    | ComputeFFT
    | SampleContracts  # NEW
    | ProcessBatch  # NEW
    | StackTensors  # NEW
    | ComputeMeanFFT  # NEW
)
```

**Interpreter Implementation**:

```python
class MonteCarloInterpreter:
    async def _sample_contracts(
        self,
        sampler_id: str,
        num_samples: int,
        output_tensor_id: str,
    ) -> Result[None, MonteCarloError]:
        """Sample contracts from Sobol sampler."""
        sampler = self._registry.get_sampler(sampler_id)

        # Sample num_samples × 5 matrix (S, K, T, r, σ)
        contracts = sampler.sample(num_samples)

        self._registry.register_tensor(output_tensor_id, contracts)
        return Success(None)

    async def _process_batch(
        self,
        contracts_tensor_id: str,
        batch_idx: int,
        config: dict,
    ) -> Result[None, MonteCarloError]:
        """Process batch of contracts (composite effect expansion).

        This method expands the high-level ProcessBatch effect into per-contract
        effect sequences and executes them.
        """
        contracts = self._registry.get_tensor(contracts_tensor_id)
        num_contracts = contracts.shape[0]

        fft_result_ids = []

        for contract_idx in range(num_contracts):
            contract = contracts[contract_idx]
            S, K, T, r, sigma = contract

            # Build per-contract effect sequence
            contract_effects = sequence_effects(
                GenerateNormals(
                    rows=config["num_paths"],
                    cols=config["num_steps"],
                    seed=config["seed"] + contract_idx,
                    output_tensor_id=f"normals_{batch_idx}_{contract_idx}",
                ),
                SimulatePaths(
                    normals_tensor_id=f"normals_{batch_idx}_{contract_idx}",
                    S=S.item(),
                    r=r.item(),
                    sigma=sigma.item(),
                    T=T.item(),
                    num_steps=config["num_steps"],
                    output_tensor_id=f"paths_{batch_idx}_{contract_idx}",
                ),
                ComputeMeanFFT(
                    input_tensor_id=f"paths_{batch_idx}_{contract_idx}",
                    output_tensor_id=f"fft_{batch_idx}_{contract_idx}",
                ),
            )

            # Execute per-contract effects
            result = await self._interpreter.interpret_sequence(contract_effects)
            if isinstance(result, Failure):
                return result

            fft_result_ids.append(f"fft_{batch_idx}_{contract_idx}")

        # Stack results
        stack_effect = StackTensors(
            input_tensor_ids=tuple(fft_result_ids),
            dim=0,
            output_tensor_id=f"batch_fft_{batch_idx}",
        )
        return await self._stack_tensors(
            stack_effect.input_tensor_ids,
            stack_effect.dim,
            stack_effect.output_tensor_id,
        )

    async def _stack_tensors(
        self,
        input_tensor_ids: tuple[str, ...],
        dim: int,
        output_tensor_id: str,
    ) -> Result[None, MonteCarloError]:
        """Stack multiple tensors."""
        tensors = [self._registry.get_tensor(tid) for tid in input_tensor_ids]
        stacked = torch.stack(tensors, dim=dim)
        self._registry.register_tensor(output_tensor_id, stacked)
        return Success(None)

    async def _compute_mean_fft(
        self,
        input_tensor_id: str,
        output_tensor_id: str,
    ) -> Result[None, MonteCarloError]:
        """Compute FFT with mean reduction."""
        paths = self._registry.get_tensor(input_tensor_id)

        # Mean reduction
        paths_mean = paths.mean(dim=0)

        # FFT
        fft_result = torch.fft.fft(paths_mean)

        self._registry.register_tensor(output_tensor_id, fft_result)
        return Success(None)
```

#### Task 2.3: Add GPU Effects

**File**: `src/spectralmc/effects/gpu.py`

```python
@dataclass(frozen=True)
class SplitInputs:
    """Request to split contract tensor into real/imaginary inputs.

    Training requires converting contract parameters (S, K, T, r, σ) into
    real and imaginary components for CVNN input.

    Attributes:
        kind: Discriminator for pattern matching. Always "SplitInputs".
        contracts_tensor_id: Identifier for the contracts tensor.
        real_output_tensor_id: Identifier for the real component output.
        imag_output_tensor_id: Identifier for the imaginary component output.
    """
    kind: Literal["SplitInputs"] = "SplitInputs"
    contracts_tensor_id: str = ""
    real_output_tensor_id: str = "real_input"
    imag_output_tensor_id: str = "imag_input"


# Update GPUEffect union
GPUEffect = (
    TensorTransfer
    | StreamSync
    | KernelLaunch
    | DLPackTransfer
    | SplitInputs  # NEW
)
```

**Interpreter Implementation**:

```python
class GPUInterpreter:
    async def _split_inputs(
        self,
        contracts_tensor_id: str,
        real_output_tensor_id: str,
        imag_output_tensor_id: str,
    ) -> Result[None, GPUError]:
        """Split contracts into real/imaginary inputs."""
        contracts = self._registry.get_tensor(contracts_tensor_id)

        # Real component: contract parameters
        real_input = contracts  # Shape: (batch_size, 5)

        # Imaginary component: zeros (initial guess for untrained model)
        imag_input = torch.zeros_like(contracts)

        self._registry.register_tensor(real_output_tensor_id, real_input)
        self._registry.register_tensor(imag_output_tensor_id, imag_input)

        return Success(None)
```

#### Task 2.4: Add Storage Effects

**File**: `src/spectralmc/effects/storage.py`

```python
@dataclass(frozen=True)
class CommitCheckpoint:
    """Request to conditionally commit checkpoint to blockchain storage.

    Blockchain commits are expensive and should only happen based on CommitPlan
    (e.g., every N steps, or on final step). This effect models conditional commits.

    Attributes:
        kind: Discriminator for pattern matching. Always "CommitCheckpoint".
        checkpoint_id: Identifier for the checkpoint to commit.
        commit_plan: CommitPlan instance (NoCommit, EveryNSteps, FinalStepOnly).
        current_step: Current training step.
        total_steps: Total training steps.
        blockchain_store_id: Identifier for the AsyncBlockchainModelStore.
    """
    kind: Literal["CommitCheckpoint"] = "CommitCheckpoint"
    checkpoint_id: str = ""
    commit_plan: str = "NoCommit"  # Serialized CommitPlan
    current_step: int = 0
    total_steps: int = 0
    blockchain_store_id: str = ""


# Update StorageEffect union
StorageEffect = (
    ReadObject
    | WriteObject
    | CommitVersion
    | CommitCheckpoint  # NEW
)
```

**Interpreter Implementation**:

```python
class StorageInterpreter:
    async def _commit_checkpoint(
        self,
        checkpoint_id: str,
        commit_plan: str,
        current_step: int,
        total_steps: int,
        blockchain_store_id: str,
    ) -> Result[None, StorageError]:
        """Conditionally commit checkpoint based on CommitPlan."""
        # Deserialize CommitPlan
        plan = self._deserialize_commit_plan(commit_plan)

        # Check if should commit
        should_commit = False
        match plan:
            case NoCommit():
                should_commit = False
            case EveryNSteps(n):
                should_commit = (current_step % n == 0)
            case FinalStepOnly():
                should_commit = (current_step == total_steps - 1)

        if not should_commit:
            return Success(None)

        # Get checkpoint and blockchain store
        checkpoint = self._registry.get_checkpoint(checkpoint_id)
        store = self._registry.get_blockchain_store(blockchain_store_id)

        # Commit to blockchain
        try:
            await store.commit_version(checkpoint)
            return Success(None)
        except Exception as e:
            return Failure(StorageError(message=f"Commit failed: {e}"))
```

#### Deliverables

- [ ] 4 new training effects implemented and tested
- [ ] 4 new Monte Carlo effects implemented and tested
- [ ] 1 new GPU effect implemented and tested
- [ ] 1 new storage effect implemented and tested
- [ ] All 11 interpreter methods implemented
- [ ] Unit tests for each new effect type
- [ ] All existing tests still pass

**Exit Criteria**: Can express complete training workflow using effect ADTs.

---

### Step 3: Effect Builder Rewrite (1 week)

**Goal**: Make `build_training_step_effects()` match actual training workflow

**Why This Step Matters**:
- Current effect builder has 11 mismatches with real implementation
- Effect builder is the blueprint for training - must be correct
- Cannot migrate `train()` until effect builder produces correct sequence

#### Task 3.1: Analyze Current Mismatches

**File**: `src/spectralmc/gbm_trainer.py:907-1024`

**Current Implementation** (11 mismatches):
1. ❌ Hardcoded market params (S=100, r=0.05) instead of Sobol sampling
2. ❌ Missing `SampleContracts` effect
3. ❌ Missing per-contract processing loop
4. ❌ Uses non-existent tensor IDs (`batch_{idx}`)
5. ❌ Missing `ZeroGrad` effect before forward pass
6. ❌ Uses single `ForwardPass` instead of `ForwardPassComplex`
7. ❌ Uses single `ComputeLoss` instead of `ComputeComplexLoss`
8. ❌ Missing `ComputeGradNorm` effect
9. ❌ Missing metadata updates for Sobol skip tracking
10. ❌ Incorrect effect sequence order
11. ❌ Missing `ProcessBatch` composite effect

**Actual Workflow** (from `_run_batch()` lines 1533-1600):
1. Sample contracts from Sobol sampler
2. For each contract:
   - Generate normals
   - Simulate GBM paths
   - Compute FFT
3. Stack FFT results into batch tensor
4. Split batch into real/imag inputs
5. Zero gradients
6. Forward pass (complex-valued)
7. Compute loss (complex-valued)
8. Backward pass
9. Compute gradient norm
10. Optimizer step
11. Log metrics
12. Update metadata (global_step, sobol_skip)

#### Task 3.2: Rewrite Effect Builder

**File**: `src/spectralmc/gbm_trainer.py:907-1024`

**Delete Current Implementation** (lines 907-1024)

**New Implementation**:

```python
def build_training_step_effects(
    self,
    batch_idx: int,
    config: TrainingConfig,
) -> Result[EffectSequence, TrainerError]:
    """Build effect sequence for single training step.

    This method produces a pure effect description matching the actual training
    workflow in _run_batch(). The effect sequence will be executed by
    SpectralMCInterpreter.

    Args:
        batch_idx: Batch index for unique tensor IDs.
        config: Training configuration.

    Returns:
        Success with EffectSequence, or Failure with TrainerError.
    """
    try:
        effects = sequence_effects(
            # 1. Sample contracts from Sobol sampler
            SampleContracts(
                sampler_id="sobol_sampler",
                num_samples=config.batch_size,
                output_tensor_id=f"contracts_{batch_idx}",
            ),

            # 2. Process batch of contracts (composite effect)
            # This expands to per-contract: GenerateNormals → SimulatePaths → ComputeFFT
            ProcessBatch(
                contracts_tensor_id=f"contracts_{batch_idx}",
                batch_idx=batch_idx,
                config={
                    "num_paths": config.num_paths,
                    "num_steps": config.num_steps,
                    "seed": config.seed + batch_idx * config.batch_size,
                },
            ),

            # 3. Split batch into real/imaginary inputs for CVNN
            SplitInputs(
                contracts_tensor_id=f"contracts_{batch_idx}",
                real_output_tensor_id=f"real_in_{batch_idx}",
                imag_output_tensor_id=f"imag_in_{batch_idx}",
            ),

            # 4. Training step
            ZeroGrad(optimizer_id="adam", set_to_none=True),

            ForwardPassComplex(
                model_id="cvnn",
                real_input_tensor_id=f"real_in_{batch_idx}",
                imag_input_tensor_id=f"imag_in_{batch_idx}",
                real_output_tensor_id=f"pred_real_{batch_idx}",
                imag_output_tensor_id=f"pred_imag_{batch_idx}",
            ),

            ComputeComplexLoss(
                pred_real_tensor_id=f"pred_real_{batch_idx}",
                pred_imag_tensor_id=f"pred_imag_{batch_idx}",
                target_tensor_id=f"targets_{batch_idx}",
                loss_type="mse",
                output_tensor_id=f"loss_{batch_idx}",
            ),

            BackwardPass(loss_tensor_id=f"loss_{batch_idx}"),

            ComputeGradNorm(
                model_id="cvnn",
                output_tensor_id=f"grad_norm_{batch_idx}",
            ),

            OptimizerStep(optimizer_id="adam"),

            StreamSync(stream_type="torch"),

            # 5. Metrics and metadata
            LogMetrics(
                metrics=(
                    ("loss", f"{{loss_{batch_idx}}}"),  # Will be resolved from registry
                    ("grad_norm", f"{{grad_norm_{batch_idx}}}"),
                ),
                step=batch_idx,
            ),

            UpdateMetadata(
                key="global_step",
                operation="increment",
            ),

            UpdateMetadata(
                key="sobol_skip",
                operation="add",
                value=config.batch_size,
            ),
        )

        return Success(effects)

    except Exception as e:
        return Failure(TrainerError(message=f"Failed to build effects: {e}"))
```

#### Task 3.3: Validation

**Test**: Compare effect sequence against `_run_batch()` line-by-line

**Validation Checklist**:
- [ ] Uses `SampleContracts` instead of hardcoded params
- [ ] Includes `ProcessBatch` for per-contract processing
- [ ] Uses correct tensor IDs matching registry expectations
- [ ] Includes `ZeroGrad` before forward pass
- [ ] Uses `ForwardPassComplex` for CVNN
- [ ] Uses `ComputeComplexLoss` for dual MSE
- [ ] Includes `ComputeGradNorm` for logging
- [ ] Updates metadata (global_step, sobol_skip)
- [ ] Effect sequence matches `_run_batch()` order exactly

#### Deliverables

- [ ] `build_training_step_effects()` rewritten (lines 907-1024)
- [ ] Effect sequence matches actual training workflow
- [ ] All effect types used are implemented (from Step 2)
- [ ] Unit test comparing effect sequence to `_run_batch()` operations
- [ ] All existing tests still pass

**Exit Criteria**: Effect builder produces correct, executable effect sequence.

---

### Step 4: Training Migration (1-2 weeks)

**Goal**: Replace imperative training with effect-based execution

**Why This Step Matters**:
- **HIGHEST RISK STEP** - Touching working code with 100% test coverage
- Core migration from Phase 2 to Phase 4
- All training logic moves from imperative to declarative

**Risk Mitigation**:
- Keep `_run_batch()` as fallback during migration
- Add integration tests before deletion
- Performance benchmark before/after
- Staged rollout: effect-based optional, then default, then mandatory

#### Task 4.1: Implement `train_via_effects()`

**File**: `src/spectralmc/gbm_trainer.py:1689-1706`

**Current Implementation**:
```python
async def train_via_effects(
    self,
    config: TrainingConfig,
    *,
    interpreter: SpectralMCInterpreter,
    logger: StepLogger | None = None,
    blockchain_store: AsyncBlockchainModelStore | None = None,
    commit_plan: CommitPlan = NoCommit(),
) -> Result[TrainingResult, TrainerError]:
    """Execute training via effect interpreter (STUB - not implemented)."""
    return Failure(TrainerError(message="train_via_effects not implemented"))
```

**New Implementation**:
```python
async def train_via_effects(
    self,
    config: TrainingConfig,
    *,
    interpreter: SpectralMCInterpreter,
    logger: StepLogger | None = None,
    blockchain_store: AsyncBlockchainModelStore | None = None,
    commit_plan: CommitPlan = NoCommit(),
) -> Result[TrainingResult, TrainerError]:
    """Execute training via effect interpreter.

    This is the Phase 4 implementation: all side effects modeled as ADTs,
    executed through a single interpreter entry point.

    Args:
        config: Training configuration.
        interpreter: SpectralMCInterpreter instance.
        logger: Optional step logger for progress tracking.
        blockchain_store: Optional blockchain storage.
        commit_plan: Commit plan for checkpoints.

    Returns:
        Success with TrainingResult, or Failure with TrainerError.
    """
    # Register models, optimizers, samplers in interpreter registry
    interpreter.registry.register_model("cvnn", self._cvnn)
    interpreter.registry.register_optimizer("adam", self._optimizer)

    # Unwrap sampler result
    match self._sampler_result:
        case Failure(error):
            return Failure(TrainerError(message=f"Sampler initialization failed: {error}"))
        case Success(sampler):
            interpreter.registry.register_sampler("sobol_sampler", sampler)

    # Register blockchain store if provided
    if blockchain_store is not None:
        interpreter.registry.register_blockchain_store("main_store", blockchain_store)

    # Build complete training effects for all batches
    all_batch_effects = []
    for batch_idx in range(config.num_batches):
        batch_effects_result = self.build_training_step_effects(batch_idx, config)

        match batch_effects_result:
            case Failure(error):
                return Failure(error)
            case Success(batch_effects):
                all_batch_effects.append(batch_effects)

        # Add checkpoint commit effect if needed
        if blockchain_store is not None:
            commit_effect = CommitCheckpoint(
                checkpoint_id=f"checkpoint_{batch_idx}",
                commit_plan=self._serialize_commit_plan(commit_plan),
                current_step=batch_idx,
                total_steps=config.num_batches,
                blockchain_store_id="main_store",
            )
            all_batch_effects.append(commit_effect)

    # Combine all batch effects into single sequence
    complete_training_effects = sequence_effects(*all_batch_effects)

    # Execute via interpreter (single entry point)
    result = await interpreter.interpret_sequence(complete_training_effects)

    match result:
        case Failure(error):
            return Failure(TrainerError(message=f"Training execution failed: {error}"))
        case Success(_):
            # Extract final metrics from registry
            final_loss_tensor = interpreter.registry.get_tensor(f"loss_{config.num_batches - 1}")
            final_grad_norm_tensor = interpreter.registry.get_tensor(f"grad_norm_{config.num_batches - 1}")

            final_loss = final_loss_tensor.item()
            final_grad_norm = final_grad_norm_tensor.item()

            # Build training result
            training_result = TrainingResult(
                final_loss=final_loss,
                final_grad_norm=final_grad_norm,
                num_batches=config.num_batches,
                total_steps=config.num_batches,
            )

            return Success(training_result)
```

#### Task 4.2: Update `train()` to Delegate

**File**: `src/spectralmc/gbm_trainer.py:1457-1687`

**Current Implementation**: 230 lines of imperative training logic with direct GPU operations

**New Implementation** (delegate to effect-based):
```python
def train(
    self,
    config: TrainingConfig,
    *,
    logger: StepLogger | None = None,
    blockchain_store: AsyncBlockchainModelStore | None = None,
    commit_plan: CommitPlan = NoCommit(),
) -> Result[TrainingResult, TrainerError]:
    """Train model using effect-based execution (Phase 4).

    This method delegates to train_via_effects(), which executes all training
    logic through the SpectralMCInterpreter. All side effects are modeled as ADTs.

    Args:
        config: Training configuration.
        logger: Optional step logger for progress tracking.
        blockchain_store: Optional blockchain storage.
        commit_plan: Commit plan for checkpoints.

    Returns:
        Success with TrainingResult, or Failure with TrainerError.
    """
    # Validate configuration
    config_validation = self._validate_config(config)
    if isinstance(config_validation, Failure):
        return config_validation

    # Create interpreter
    interpreter = SpectralMCInterpreter.create(
        torch_stream=self._context.torch_stream,
        cupy_stream=self._context.cupy_stream if self._context.cupy_stream else None,
        storage_bucket=blockchain_store.bucket if blockchain_store else "",
    )

    # Delegate to effect-based implementation
    return asyncio.run(self.train_via_effects(
        config,
        interpreter=interpreter,
        logger=logger,
        blockchain_store=blockchain_store,
        commit_plan=commit_plan,
    ))
```

**Impact**: Reduces `train()` from 230 lines of imperative code to ~40 lines of pure delegation.

#### Task 4.3: Delete Imperative Methods

**CRITICAL**: Only delete after `train_via_effects()` integration tests pass.

**Deletions**:

1. **Delete `_run_batch()`** (lines 1533-1600)
   - 67 lines of direct GPU operations
   - Direct `.backward()`, `.step()` calls
   - Direct CUDA kernel launches for FFT

2. **Delete `_torch_step()`** (lines 820-836)
   - 16 lines of direct PyTorch training step
   - Direct `optimizer.zero_grad()`, `optimizer.step()`

3. **Delete `_simulate_fft()`** direct GPU usage (lines 807-819)
   - 12 lines of direct CuPy FFT calls
   - Should be replaced by `ComputeMeanFFT` effect

**Before Deletion Checklist**:
- [ ] `train_via_effects()` fully implemented
- [ ] Integration tests pass (same results as imperative)
- [ ] Performance benchmark shows <10% overhead
- [ ] All 361+ tests still pass with new implementation

#### Task 4.4: Add Integration Tests

**File**: `tests/test_gbm_trainer.py` (add new tests)

**Test Coverage Required**:

```python
@pytest.mark.asyncio
async def test_train_via_effects_produces_same_results_as_imperative():
    """Test effect-based training produces identical results to imperative."""
    config = TrainingConfig(
        num_batches=10,
        batch_size=32,
        num_paths=1000,
        num_steps=252,
        seed=42,
    )

    # Train with imperative implementation (before deletion)
    trainer_imperative = GBMTrainer(...)
    result_imperative = trainer_imperative.train(config)

    # Train with effect-based implementation
    trainer_effects = GBMTrainer(...)
    result_effects = asyncio.run(trainer_effects.train_via_effects(
        config,
        interpreter=SpectralMCInterpreter.create(...),
    ))

    # Compare results
    assert isinstance(result_imperative, Success)
    assert isinstance(result_effects, Success)

    # Same final loss (within floating point tolerance)
    assert abs(result_imperative.value.final_loss - result_effects.value.final_loss) < 1e-5

    # Same final grad norm
    assert abs(result_imperative.value.final_grad_norm - result_effects.value.final_grad_norm) < 1e-5


@pytest.mark.asyncio
async def test_train_via_effects_performance_overhead():
    """Test effect system overhead is acceptable (<10%)."""
    import time

    config = TrainingConfig(num_batches=100, batch_size=32)

    # Benchmark imperative
    trainer_imperative = GBMTrainer(...)
    start = time.time()
    trainer_imperative.train(config)
    imperative_time = time.time() - start

    # Benchmark effect-based
    trainer_effects = GBMTrainer(...)
    start = time.time()
    asyncio.run(trainer_effects.train_via_effects(
        config,
        interpreter=SpectralMCInterpreter.create(...),
    ))
    effects_time = time.time() - start

    # Verify overhead <10%
    overhead = (effects_time - imperative_time) / imperative_time
    assert overhead < 0.10, f"Effect system overhead {overhead:.1%} exceeds 10%"


@pytest.mark.asyncio
async def test_train_via_effects_blockchain_commits():
    """Test blockchain commits work via effects."""
    config = TrainingConfig(num_batches=10)
    blockchain_store = AsyncBlockchainModelStore(...)
    commit_plan = EveryNSteps(n=5)

    trainer = GBMTrainer(...)
    result = await trainer.train_via_effects(
        config,
        interpreter=SpectralMCInterpreter.create(...),
        blockchain_store=blockchain_store,
        commit_plan=commit_plan,
    )

    assert isinstance(result, Success)

    # Verify commits happened
    chain = await blockchain_store.get_chain()
    assert len(chain.versions) == 2  # Steps 5 and 10
```

#### Deliverables

- [ ] `train_via_effects()` fully implemented (lines 1689-1706 → ~100 lines)
- [ ] `train()` updated to delegate (lines 1457-1687 → ~40 lines)
- [ ] `_run_batch()` deleted (lines 1533-1600)
- [ ] `_torch_step()` deleted (lines 820-836)
- [ ] `_simulate_fft()` direct GPU usage deleted (lines 807-819)
- [ ] Integration tests added (3+ new tests)
- [ ] Performance benchmark shows <10% overhead
- [ ] All 361+ tests pass with effect-based implementation

**Exit Criteria**: `train()` uses effects exclusively, imperative methods deleted, all tests pass.

---

### Step 5: Logging Migration (1-2 days)

**Goal**: Complete logging migration per Phase 4 requirement

**Why This Step Matters**:
- Phase 4 requires "All side effects modeled as ADTs"
- Logging is a side effect
- 28 direct `logger.*()` calls in Tier 3 storage code violate requirement

#### Task 5.1: Training Logging (Already Done)

**Status**: ✅ Training logging already uses `LogMetrics` effects via effect builder (Step 3).

**No Action Required**: Effect builder (Step 3) includes `LogMetrics` effects.

#### Task 5.2: Storage Logging Migration

**Files to Modify** (28 direct logger calls):
1. `src/spectralmc/storage/tensorboard_writer.py` - 10 calls
2. `src/spectralmc/storage/inference.py` - 15 calls
3. `src/spectralmc/storage/store.py` - 2 calls
4. `src/spectralmc/storage/gc.py` - 1 call

**Migration Pattern**:

**Before**:
```python
import logging
logger = logging.getLogger(__name__)

def some_method():
    logger.info(f"Starting operation: {param}")
    # ... do work ...
    logger.error(f"Operation failed: {error}")
```

**After**:
```python
from spectralmc.effects import LogMessage, LoggingInterpreter

def some_method(logging_interpreter: LoggingInterpreter):
    # Log start
    log_effect = LogMessage(level="info", message=f"Starting operation: {param}")
    await logging_interpreter.interpret(log_effect)

    # ... do work ...

    # Log error
    error_effect = LogMessage(level="error", message=f"Operation failed: {error}")
    await logging_interpreter.interpret(error_effect)
```

**Example Migration**:

**File**: `src/spectralmc/storage/tensorboard_writer.py`

**Before** (10 logger calls):
```python
class TensorBoardWriter:
    def __init__(self, log_dir: Path):
        self._logger = logging.getLogger(__name__)

    def log_scalar(self, tag: str, value: float, step: int):
        self._logger.info(f"Logging scalar: {tag}={value} at step {step}")
        # ... TensorBoard write ...
```

**After**:
```python
class TensorBoardWriter:
    def __init__(self, log_dir: Path, logging_interpreter: LoggingInterpreter):
        self._logging_interpreter = logging_interpreter

    async def log_scalar(self, tag: str, value: float, step: int):
        log_effect = LogMessage(
            level="info",
            message=f"Logging scalar: {tag}={value} at step {step}",
        )
        await self._logging_interpreter.interpret(log_effect)
        # ... TensorBoard write ...
```

**Migration Checklist**:
- [ ] `src/spectralmc/storage/tensorboard_writer.py` - Convert 10 logger calls
- [ ] `src/spectralmc/storage/inference.py` - Convert 15 logger calls
- [ ] `src/spectralmc/storage/store.py` - Convert 2 logger calls
- [ ] `src/spectralmc/storage/gc.py` - Convert 1 logger call
- [ ] Update all call sites to pass `LoggingInterpreter` instance
- [ ] Add tests for logging effects in storage layer

#### Deliverables

- [ ] All 28 logger calls converted to `LogMessage` effects
- [ ] Storage classes accept `LoggingInterpreter` dependency
- [ ] Tests verify logging effects are created correctly
- [ ] All existing tests still pass

**Exit Criteria**: Zero direct `logger.*()` calls in Tier 3 storage code. All logging via effects.

---

## Timeline and Milestones

| Step | Duration | Cumulative | Milestone |
|------|----------|------------|-----------|
| **Step 1: Foundation** | 2-3 days | 3 days | ✅ Effect system proven to work |
| **Step 2: Architecture** | 1-2 weeks | ~2.5 weeks | ✅ All 11 effect types implemented |
| **Step 3: Effect Builder** | 1 week | ~3.5 weeks | ✅ Effect builder matches training workflow |
| **Step 4: Training Migration** | 1-2 weeks | ~5 weeks | ✅ Training uses effects exclusively |
| **Step 5: Logging** | 1-2 days | ~5.5 weeks | ✅ All logging via effects |

**Total Estimated Effort**: 5-6 weeks

**Milestones**:
- **Week 1**: Effect system integration tests passing, interpreter bug fixed
- **Week 2-3**: All 11 new effects implemented and tested
- **Week 4**: Effect builder rewritten and validated
- **Week 5-6**: Training migration complete, imperative code deleted
- **Week 6**: Logging migration complete, Phase 4 achieved

---

## Success Criteria

Phase 4 is complete when ALL of the following are true:

### Code Quality
- ✅ **All side effects modeled as ADTs** - Zero direct GPU operations in training orchestration
- ✅ **Single interpreter entry point** - `train()` delegates to `train_via_effects()`
- ✅ **Pure code separated from effectful execution** - Effect builders are pure, interpreters execute
- ✅ **Direct side-effect APIs deprecated** - `_run_batch()`, `_torch_step()`, `_simulate_fft()` deleted
- ✅ **All logging via effects** - Zero direct `logger.*()` calls in Tier 3 code

### Testing
- ✅ **All 361+ tests passing** - No regressions from migration
- ✅ **Integration tests added** - Effect system proven to work (10-15 new tests)
- ✅ **Same results as imperative** - Effect-based training produces identical outputs
- ✅ **Performance within 10%** - Effect system overhead acceptable

### Documentation
- ✅ **Effect types documented** - All 11 new effects have docstrings
- ✅ **Migration complete** - This document marked as "COMPLETE"
- ✅ **Phase 4 achieved** - `effect_interpreter.md` updated to reflect Phase 4 status

### Architecture
- ✅ **Effect system complete** - All 11 effect types implemented
- ✅ **Interpreters complete** - All 11 interpreter methods implemented
- ✅ **Registry tested** - SharedRegistry data flow validated
- ✅ **Zero technical debt** - No temporary workarounds or TODOs

---

## Risk Mitigation

### High-Risk Areas

**Step 4 (Training Migration)** - HIGHEST RISK
- **Risk**: Touching working code with 100% test coverage
- **Impact**: Could break training, cause test failures, performance regression
- **Probability**: Medium-High (complex imperative → declarative refactor)

**Mitigation Strategies**:
1. **Complete Steps 1-3 first** - Prove foundation works before touching training code
2. **Keep imperative code as fallback** - Don't delete `_run_batch()` until integration tests pass
3. **Add integration tests before deletion** - Verify effect-based produces same results
4. **Performance benchmark** - Ensure <10% overhead before committing
5. **Staged rollout**:
   - Stage 1: `train_via_effects()` implemented, optional (both implementations exist)
   - Stage 2: `train()` delegates to `train_via_effects()`, imperative methods kept
   - Stage 3: Integration tests pass, performance validated
   - Stage 4: Delete imperative methods (`_run_batch()`, `_torch_step()`, `_simulate_fft()`)

### Fallback Plan

**If Step 4 fails** (integration tests fail, performance regression >10%, critical bugs):
- Keep both implementations (imperative and effect-based)
- Mark as **Phase 3** instead of Phase 4
- Document why migration stopped
- Revisit in future with lessons learned

**If Step 2 takes longer than expected**:
- Focus on critical effects first (ForwardPassComplex, ComputeComplexLoss, ZeroGrad, ProcessBatch)
- Defer optional effects (UpdateLearningRate)
- Adjust timeline, notify stakeholders

### Rollback Plan

**If migration causes critical failures**:
1. Git revert to last known good state
2. Re-enable imperative implementation
3. Post-mortem analysis of failure
4. Update plan with findings
5. Retry with adjusted approach

---

## Files to Modify

### New Files (1)
- `tests/test_effects/test_integration.py` - Integration tests for SpectralMCInterpreter

### Modified Files (11)

**Effect Types** (4 files):
- `src/spectralmc/effects/training.py` - Add 4 new effect types
- `src/spectralmc/effects/montecarlo.py` - Add 4 new effect types
- `src/spectralmc/effects/gpu.py` - Add 1 new effect type
- `src/spectralmc/effects/storage.py` - Add 1 new effect type

**Interpreters** (1 file):
- `src/spectralmc/effects/interpreter.py` - Add 11 interpreter methods, fix 1 bug

**Training** (1 file):
- `src/spectralmc/gbm_trainer.py` - Rewrite effect builder, migrate `train()`, delete 3 methods

**Storage Logging** (4 files):
- `src/spectralmc/storage/tensorboard_writer.py` - Convert 10 logger calls
- `src/spectralmc/storage/inference.py` - Convert 15 logger calls
- `src/spectralmc/storage/store.py` - Convert 2 logger calls
- `src/spectralmc/storage/gc.py` - Convert 1 logger call

**Exports** (1 file):
- `src/spectralmc/effects/__init__.py` - Add new effect types to `__all__`

**Total**: 1 new file, 11 modified files, ~20 specific changes

---

## Technical Deep Dives

### Why Can't Effect ADTs Express Loops?

**Problem**: Purity doctrine forbids `for` loops in business logic (Tier 2). Effect ADTs are Tier 2 code.

**Example** (forbidden):
```python
@dataclass(frozen=True)
class ProcessBatch:
    contracts: torch.Tensor  # IMPURE - mutable tensor

    def execute(self):
        for contract in self.contracts:  # FORBIDDEN - for loop in Tier 2
            # ... process contract ...
```

**Solution**: Composite effects that interpreters expand:
```python
@dataclass(frozen=True)
class ProcessBatch:
    """High-level effect that interpreter expands into per-contract sequence."""
    contracts_tensor_id: str  # PURE - just an ID
    batch_idx: int
    config: dict


# Interpreter expands composite effect
class MonteCarloInterpreter:
    async def _process_batch(self, ...):
        contracts = self._registry.get_tensor(contracts_tensor_id)

        # Interpreter can use loops (Tier 3, not Tier 2)
        for contract_idx in range(len(contracts)):
            # Build per-contract effects
            effects = sequence_effects(...)
            await self.interpret_sequence(effects)
```

**Key Insight**: Effect ADTs describe WHAT (pure), interpreters decide HOW (impure loops allowed in Tier 3).

### Why Separate `ZeroGrad` from `OptimizerStep`?

**Problem**: Current `OptimizerStep` calls `zero_grad()` AFTER `step()`, which is wrong.

**PyTorch Training Loop**:
```python
for batch in dataloader:
    optimizer.zero_grad()  # 1. Zero gradients FIRST
    output = model(input)  # 2. Forward pass
    loss = criterion(output, target)  # 3. Compute loss
    loss.backward()  # 4. Compute gradients
    optimizer.step()  # 5. Update parameters
```

**Current Bug**:
```python
OptimizerStep:
    optimizer.step()  # 5. Update parameters
    optimizer.zero_grad()  # BUG: Too late! Should be at step 1
```

**Solution**: Separate effects match PyTorch semantics:
```python
sequence_effects(
    ZeroGrad(optimizer_id="adam"),  # 1. Zero gradients
    ForwardPass(...),  # 2. Forward
    ComputeLoss(...),  # 3. Loss
    BackwardPass(...),  # 4. Gradients
    OptimizerStep(optimizer_id="adam"),  # 5. Update
)
```

**Benefit**: Effect sequence matches PyTorch training loop exactly.

### Why `ForwardPassComplex` Instead of Regular `ForwardPass`?

**Problem**: CVNN (Complex-Valued Neural Networks) require dual inputs/outputs.

**Regular Neural Network**:
```python
# Single tensor input → Single tensor output
output = model(input)
```

**Complex-Valued Neural Network**:
```python
# Dual tensor inputs → Dual tensor outputs
pred_real, pred_imag = model(real_input, imag_input)
```

**Effect Types**:
```python
# Regular forward pass (single tensor)
ForwardPass:
    input_tensor_id: str
    output_tensor_id: str

# Complex forward pass (dual tensors)
ForwardPassComplex:
    real_input_tensor_id: str
    imag_input_tensor_id: str
    real_output_tensor_id: str
    imag_output_tensor_id: str
```

**Why Not Reuse `ForwardPass`?**: Type safety - `ForwardPassComplex` explicitly models dual inputs/outputs, preventing accidental single-tensor usage with CVNN models.

---

## Next Steps After Migration

Once Phase 4 is complete, consider:

1. **Performance Optimization**:
   - Profile effect system overhead
   - Optimize hot paths in interpreters
   - Consider effect batching for efficiency

2. **Effect System Extensions**:
   - Distributed training effects (multi-GPU, multi-node)
   - Hyperparameter tuning effects
   - Model ensemble effects

3. **Reproducibility Enhancements**:
   - Serialize complete effect sequences
   - Replay training from effect logs
   - Deterministic replay guarantees

4. **Testing Improvements**:
   - Property-based testing with Hypothesis
   - Fuzzing effect sequences
   - Chaos engineering for fault injection

5. **Documentation**:
   - Effect system tutorial
   - Migration case study
   - Performance benchmarks

---

## Appendix: Command Reference

### Running Tests
```bash
# All tests
docker compose -f docker/docker-compose.yml exec spectralmc poetry run test-all

# Specific test file
docker compose -f docker/docker-compose.yml exec spectralmc poetry run test-all tests/test_effects/test_integration.py

# With coverage
docker compose -f docker/docker-compose.yml exec spectralmc poetry run pytest --cov=src/spectralmc tests/
```

### Code Quality
```bash
# Full check (ruff, black, mypy)
docker compose -f docker/docker-compose.yml exec spectralmc poetry run check-code

# Purity check
docker compose -f docker/docker-compose.yml exec spectralmc poetry run check-purity
```

### Git Workflow
```bash
# Check status
git status

# Review changes
git diff

# View commit history
git log --oneline

# Create feature branch (user only - not Claude)
git checkout -b feature/effect-migration-step-1

# Commit changes (user only - not Claude)
git add .
git commit -m "Step 1: Add effect system integration tests"
```

---

## Document History

- **2026-01-03**: Initial creation - Migration plan for Phase 2 → Phase 4
- **Status**: ACTIVE - Migration in progress
- **Next Review**: After Step 1 completion (foundation validation)

---

**End of Document**
