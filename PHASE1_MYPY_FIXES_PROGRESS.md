# Phase 1 Mypy Fixes Progress Report

## Session Date
2025-11-06

## Objective
Fix mypy strict type errors discovered after implementing Sub-Phases 1A and 1B.

## Completed Work

### 1. Documentation Updates
- ✅ Added type safety requirements to `SPECTRALMC_COMPLETION_PLAN.md` (lines 231-252)
- ✅ Added "Type Safety" section to `CLAUDE.md` with strict requirements
- ✅ Documented 25 remaining explicit Any errors for future work

### 2. Critical Type Errors Fixed (3 errors)

#### Error 1: `gbm.py:204` - ConcurrentNormGenerator API change
**Fix**: Changed to use `BufferConfig.create()` instead of passing int directly
```python
# Before:
self._ngen = ConcurrentNormGenerator(self._sp.buffer_size, ngen_cfg)

# After:
buffer_cfg = BufferConfig.create(self._sp.buffer_size, ngen_cfg.rows, ngen_cfg.cols)
self._ngen = ConcurrentNormGenerator(buffer_cfg, ngen_cfg)
```

#### Error 2: `gbm_trainer.py:217` - Type narrowing for AnyDType
**Fix**: Added isinstance check to narrow from AnyDType to FullPrecisionDType
```python
dtype_any: AnyDType
self._device, dtype_any = module_state_device_dtype(self._cvnn.state_dict())

assert isinstance(dtype_any, FullPrecisionDType), (
    f"CVNN must use full precision dtype (float32/64, complex64/128), got {dtype_any}"
)
self._dtype: FullPrecisionDType = dtype_any
```

#### Error 3: `gbm_trainer.py:241` - SobolSampler API change
**Fix**: Changed to use `SobolConfig` instead of seed/skip kwargs
```python
# Before:
self._sampler = SobolSampler(..., seed=..., skip=...)

# After:
self._sampler = SobolSampler(
    ...,
    config=SobolConfig(seed=self._sim_params.mc_seed, skip=self._sobol_skip),
)
```

### 3. Test Fixes

#### `tests/test_gbm.py`
- Added `SobolConfig` import
- Updated `test_black_scholes_mc` to use new API:
```python
sampler = SobolSampler(
    pydantic_class=BlackScholes.Inputs,
    dimensions=_BS_DIMENSIONS,
    config=SobolConfig(seed=31, skip=0),
)
```

### 4. Optimizer State Refactoring

#### Changed type from `Mapping[str, object]` to `AdamOptimizerState`
**Files Modified**:
- `src/spectralmc/gbm_trainer.py`:
  - Line 72-77: Added `AdamOptimizerState` import
  - Line 130: Changed field type to `Optional[AdamOptimizerState]`
  - Line 215: Changed instance variable type
  - Line 338: Use `optimizer_state.to_torch()` when loading
  - Line 388-396: Convert state_dict to CPU and wrap in `AdamOptimizerState.from_torch()`

**Reason**: `AdamOptimizerState` is a structured Pydantic model that enforces:
- All tensors on CPU (required for serialization)
- Complete type safety
- Proper validation

### 5. Sobol Sampler Skip Tracking

**Problem**: Input contract generation wasn't deterministic across snapshots.

**Solution**: Added separate tracking for Sobol sampler position:
- Added `sobol_skip: int = 0` field to `GbmCVNNPricerConfig` (line 132)
- Track in instance: `self._sobol_skip` (line 218)
- Initialize sampler with tracked skip (line 258)
- Increment after each sample: `self._sobol_skip += batch_size` (line 351)
- Include in snapshot (line 284)

**Rationale**: `sim_params.skip` is for MC engine's normal generator, not for input contracts. These need separate tracking.

## Test Results

**Status**: 14 out of 16 tests passing ✅

### Passing Tests (14)
- `test_gbm.py::test_black_scholes_mc[float32]` ✅
- `test_gbm.py::test_black_scholes_mc[float64]` ✅
- `test_gbm.py::test_multi_price[float32]` ✅
- `test_gbm.py::test_multi_price[float64]` ✅
- `test_gbm_trainer.py::test_lockstep_training[float32]` ✅
- `test_gbm_trainer.py::test_lockstep_training[float64]` ✅
- `test_gbm_trainer.py::test_snapshot_cycle_deterministic[float32]` ✅
- `test_gbm_trainer.py::test_snapshot_cycle_deterministic[float64]` ✅
- `test_gbm_trainer.py::test_snapshot_optimizer_serialization_roundtrip[float32]` ✅
- `test_gbm_trainer.py::test_snapshot_optimizer_serialization_roundtrip[float64]` ✅
- `test_gbm_trainer.py::test_predict_price_smoke[float32]` ✅
- `test_gbm_trainer.py::test_predict_price_smoke[float64]` ✅
- `test_gbm_trainer.py::test_gbm_trainer_smoke[float32]` ✅
- `test_gbm_trainer.py::test_gbm_trainer_smoke[float64]` ✅

### Failing Tests (2)
- `test_gbm_trainer.py::test_snapshot_restart_without_optimizer[float32]` ❌
- `test_gbm_trainer.py::test_snapshot_restart_without_optimizer[float64]` ❌

**Error**:
```
assert _max_param_diff(trainer._cvnn, restarted._cvnn) == 0.0
AssertionError: assert 0.03177034854888916 == 0.0  (float32)
AssertionError: assert 0.03296413114457725 == 0.0  (float64)
```

**Test Scenario**:
```python
trainer.train(num_batches=3, batch_size=8, learning_rate=LEARNING_RATE)
snap = trainer.snapshot().model_copy(update={"optimizer_state": None, "cvnn": _clone_model(trainer._cvnn)})
restarted = GbmCVNNPricer(snap)

trainer.train(num_batches=2, batch_size=8, learning_rate=LEARNING_RATE)
restarted.train(num_batches=2, batch_size=8, learning_rate=LEARNING_RATE)

# Expects models to be identical, but they diverge
assert _max_param_diff(trainer._cvnn, restarted._cvnn) == 0.0
```

## Remaining Work

### Immediate Issue: test_snapshot_restart_without_optimizer

**Analysis**:
1. Test sets `optimizer_state=None` explicitly
2. Both trainers continue training with fresh Adam optimizers
3. Fresh optimizers start with zero momentum/variance state
4. This should cause divergence in gradient updates
5. **Question**: Is the test expectation correct?

**Possible Root Causes**:
1. **Torch random state not captured**: `torch.manual_seed(seed)` in test setup, but internal CUDA random state might differ
2. **MC engine state divergence**: The MC engine's RNG state is saved, but may not be perfectly synchronized
3. **Expected behavior**: Without optimizer state, Adam's momentum buffers are reset, causing different updates

**Next Steps**:
1. Check if this test passed before Sub-Phase 1B changes
2. Verify if `_clone_model` properly resets all state
3. Investigate if torch random state needs to be included in snapshot
4. Consider if test expectation is realistic (perfect determinism without optimizer state)

### Remaining Type Errors (25 explicit Any)

**Not blocking, documented for future work**:

Files with explicit `Any` annotations to fix:
- `sobol_sampler.py` - lines 50, 71 (2 errors)
- `async_normals.py` - lines 86, 135 (2 errors)
- `gbm.py` - lines 51, 56, 90, 153, 163, 170, 178 (7 errors)
- `models/torch.py` - lines 243, 314, 345, 405, 431 (5 errors)
- `cvnn_factory.py` - lines 53, 57, 64, 73, 87, 93, 100 (7 errors)
- `gbm_trainer.py` - lines 121, 133 (2 errors)

Most are in Pydantic `arbitrary_types_allowed` configs and can use `object` instead of `Any`.

## Files Modified This Session

1. `SPECTRALMC_COMPLETION_PLAN.md` - Added type safety status
2. `CLAUDE.md` - Added type safety section
3. `src/spectralmc/gbm.py` - Added BufferConfig import and usage
4. `src/spectralmc/gbm_trainer.py` - Multiple changes for AdamOptimizerState and sobol_skip
5. `tests/test_gbm.py` - Updated to use SobolConfig

## Commands to Resume

```bash
# Check current test status
docker compose -f docker/docker-compose.yml exec spectralmc pytest tests/test_gbm_trainer.py::test_snapshot_restart_without_optimizer -v

# Run mypy to verify no regressions
docker compose -f docker/docker-compose.yml exec spectralmc mypy src/spectralmc --strict --disallow-any-explicit > /tmp/mypy-current.txt 2>&1

# Run full test suite
docker compose -f docker/docker-compose.yml exec spectralmc poetry run test-all
```

## Decision Needed

**Question for user**: Should `test_snapshot_restart_without_optimizer` expect perfect determinism?

**Context**:
- When `optimizer_state=None`, a fresh Adam optimizer is created with zero momentum/variance
- This is different from continuing with existing momentum, so gradients will differ
- The test expects `_max_param_diff == 0.0` (perfect match)
- Current divergence: ~0.03 (3% relative difference)

**Options**:
1. **Relax test tolerance**: Accept small differences when optimizer state is missing
2. **Fix determinism**: Investigate if there's a torch/CUDA random state we're not capturing
3. **Test is incorrect**: Maybe test should verify convergence quality, not exact match
