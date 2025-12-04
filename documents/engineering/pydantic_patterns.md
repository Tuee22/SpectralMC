# Pydantic Best Practices

## Overview

SpectralMC uses **Pydantic v2** for all configuration and data validation. Pydantic models provide runtime type validation, immutable configurations, and seamless integration with mypy strict mode.

**Related Standards**: [Coding Standards](coding_standards.md), [Documentation Standards](../documentation_standards.md)

---

## Model Definition

Use Pydantic `BaseModel` for all configuration objects and data validation:

```python
from pydantic import BaseModel, ConfigDict, model_validator

class BoundSpec(BaseModel):
    """Inclusive numeric bounds for a single coordinate axis."""

    lower: float
    upper: float

    @model_validator(mode="after")
    def _validate(self) -> "BoundSpec":
        """Ensure the lower bound is strictly less than the upper bound."""
        if self.lower >= self.upper:
            raise ValueError("`lower` must be strictly less than `upper`.")
        return self
```

**Key features**:
- Type-annotated fields (`lower: float`, `upper: float`)
- Runtime validation via `@model_validator`
- Docstrings on class and validators
- Explicit return type on validators

---

## Configuration Classes

Use `ConfigDict` to configure Pydantic model behavior. SpectralMC requires **strict configuration**:

### Required Config: `extra="forbid"`

All Pydantic models **must** forbid extra fields to prevent typos and silent configuration errors:

```python
from pydantic import BaseModel, ConfigDict

class BlackScholesConfig(BaseModel):
    """Configuration for Black-Scholes Monte Carlo simulation."""

    model_config = ConfigDict(
        extra="forbid",  # REQUIRED: Reject unknown fields
        frozen=True,     # RECOMMENDED: Make immutable
    )

    spot: float
    rate: float
    volatility: float
    maturity: float
```

**Why `extra="forbid"`?**

```python
# ✅ CORRECT - Raises ValidationError
config = BlackScholesConfig(
    spot=100.0,
    rate=0.05,
    volatility=0.2,
    maturity=1.0,
    volatlity=0.3  # Typo! Pydantic catches this
)
# ValidationError: Extra inputs are not permitted

# ❌ WITHOUT extra="forbid" - Silent bug!
# Typo would be silently ignored, using wrong volatility
```

### Recommended Config: `frozen=True`

Make configuration immutable after creation:

```python
model_config = ConfigDict(
    extra="forbid",
    frozen=True,  # Prevents accidental mutation
)
```

**Benefits**:
- Thread-safe sharing across processes (Ray, Dask)
- Prevents accidental modification during training
- Clear separation between configuration and state

---

## Model Validators

Use `@model_validator(mode="after")` for cross-field validation:

```python
from pydantic import BaseModel, ConfigDict, model_validator

class BlackScholesConfig(BaseModel):
    """Configuration for Black-Scholes Monte Carlo simulation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    spot: float
    rate: float
    volatility: float
    maturity: float

    @model_validator(mode="after")
    def _validate_positive_values(self) -> "BlackScholesConfig":
        """Ensure all financial parameters are positive."""
        for field_name, value in [
            ("spot", self.spot),
            ("volatility", self.volatility),
            ("maturity", self.maturity),
        ]:
            if value <= 0:
                raise ValueError(f"{field_name} must be positive, got {value}")
        return self
```

**Patterns**:
- `mode="after"` runs after field-level validation
- Return type must be `Self` or the class name
- Raise `ValueError` with descriptive messages
- Validate business logic, not types (types validated automatically)

---

## Type Safety with Pydantic

Pydantic models integrate seamlessly with mypy. Always use **explicit type annotations**:

```python
from typing import Dict, Type

# ✅ CORRECT - Explicit typing
def create_sampler(
    bounds: Dict[str, BoundSpec],
    model_class: Type[PointT]
) -> SobolSampler[PointT]:
    return SobolSampler(bounds=bounds, model_class=model_class)

# ❌ INCORRECT - Implicit typing
def create_sampler(bounds, model_class):
    return SobolSampler(bounds=bounds, model_class=model_class)
```

**mypy plugin configuration** (`pyproject.toml`):

```toml
[tool.mypy]
plugins = ["pydantic.mypy"]

[tool.pydantic-mypy]
init_forbid_extra = true      # Enforce extra="forbid"
init_typed = true              # Require __init__ type hints
warn_required_dynamic_aliases = true
```

See [Coding Standards](coding_standards.md) for complete mypy configuration.

---

## Field Validators

Use `@field_validator` for single-field validation:

```python
from pydantic import BaseModel, field_validator

class MonteCarloConfig(BaseModel):
    """Monte Carlo simulation configuration."""

    num_paths: int
    num_steps: int
    seed: int

    @field_validator("num_paths", "num_steps")
    @classmethod
    def _validate_positive(cls, v: int, info: ValidationInfo) -> int:
        """Ensure simulation parameters are positive."""
        if v <= 0:
            raise ValueError(f"{info.field_name} must be positive, got {v}")
        return v

    @field_validator("seed")
    @classmethod
    def _validate_seed(cls, v: int) -> int:
        """Ensure seed is non-negative."""
        if v < 0:
            raise ValueError(f"seed must be non-negative, got {v}")
        return v
```

**Note**: Field validators must be `@classmethod` in Pydantic v2.

---

## Nested Models

Pydantic supports nested validation automatically:

```python
class SimulationConfig(BaseModel):
    """Complete simulation configuration."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    bounds: Dict[str, BoundSpec]  # Nested Pydantic models
    monte_carlo: MonteCarloConfig  # Nested Pydantic model
    black_scholes: BlackScholesConfig  # Nested Pydantic model

# Automatic nested validation
config = SimulationConfig(
    bounds={
        "x": BoundSpec(lower=0.0, upper=1.0),
        "y": BoundSpec(lower=-1.0, upper=1.0),
    },
    monte_carlo=MonteCarloConfig(
        num_paths=10000,
        num_steps=252,
        seed=42,
    ),
    black_scholes=BlackScholesConfig(
        spot=100.0,
        rate=0.05,
        volatility=0.2,
        maturity=1.0,
    ),
)
```

Pydantic validates:
- All nested `BoundSpec` instances
- All fields in `MonteCarloConfig`
- All fields in `BlackScholesConfig`
- Rejects any extra fields at any level

---

## Serialization

Pydantic models serialize to/from JSON and dictionaries:

```python
# To dictionary
config_dict = config.model_dump()

# To JSON
config_json = config.model_dump_json()

# From dictionary
config = SimulationConfig.model_validate(config_dict)

# From JSON
config = SimulationConfig.model_validate_json(config_json)
```

**Note**: Use `model_dump()` and `model_validate()` (Pydantic v2 API), not deprecated `dict()` and `parse_obj()`.

---

## Common Patterns

### Optional Fields with Defaults

```python
from typing import Optional

class TrainingConfig(BaseModel):
    """Training configuration."""

    learning_rate: float = 0.001  # Default value
    max_epochs: int = 100
    early_stopping: bool = True
    checkpoint_dir: Optional[str] = None  # Optional field
```

### Computed Fields

```python
from pydantic import computed_field

class GBMConfig(BaseModel):
    """Geometric Brownian Motion configuration."""

    spot: float
    rate: float
    volatility: float
    maturity: float

    @computed_field
    @property
    def drift(self) -> float:
        """Compute drift coefficient."""
        return self.rate - 0.5 * self.volatility ** 2
```

### Custom Types

```python
from pydantic import BaseModel, field_validator
from pathlib import Path

class FileConfig(BaseModel):
    """Configuration with file path validation."""

    output_path: Path

    @field_validator("output_path")
    @classmethod
    def _validate_path_exists(cls, v: Path) -> Path:
        """Ensure parent directory exists."""
        if not v.parent.exists():
            raise ValueError(f"Parent directory does not exist: {v.parent}")
        return v
```

---

## Integration with SpectralMC

### Configuration Objects

All SpectralMC configuration classes use Pydantic:

```python
# From src/spectralmc/gbm_trainer.py
class GbmCVNNPricerConfig(BaseModel):
    """Complete configuration for GBM CVNN pricer."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    # Simulation parameters
    spot: float
    rate: float
    volatility: float
    maturity: float
    num_paths: int

    # Model architecture
    hidden_dims: List[int]
    activation: str

    # Training parameters
    learning_rate: float
    batch_size: int
```

### Validation in Factories

Pydantic models validate inputs before expensive operations:

```python
def train_model(config: GbmCVNNPricerConfig) -> TrainedModel:
    """Train model with validated configuration."""
    # Pydantic already validated:
    # - All fields present
    # - Correct types
    # - Business logic constraints
    # Safe to proceed with training
    ...
```

---

## Testing Pydantic Models

Test validation logic explicitly:

```python
import pytest
from pydantic import ValidationError

def test_bound_spec_validation():
    """Test BoundSpec rejects invalid bounds."""
    # Valid bounds
    bounds = BoundSpec(lower=0.0, upper=1.0)
    assert bounds.lower == 0.0
    assert bounds.upper == 1.0

    # Invalid bounds (lower >= upper)
    with pytest.raises(ValidationError, match="lower.*less than.*upper"):
        BoundSpec(lower=1.0, upper=0.0)

def test_config_forbids_extra_fields():
    """Test configuration rejects unknown fields."""
    with pytest.raises(ValidationError, match="Extra inputs"):
        BlackScholesConfig(
            spot=100.0,
            rate=0.05,
            volatility=0.2,
            maturity=1.0,
            unknown_field=42,  # Should be rejected
        )
```

---

## Summary

- **Use Pydantic** for all configuration and data validation
- **Required config**: `extra="forbid"` on all models
- **Recommended config**: `frozen=True` for immutability
- **Validators**: Use `@model_validator` for cross-field logic
- **Type safety**: Explicit type hints on all fields
- **Nested models**: Automatic validation of nested structures
- **Testing**: Explicitly test validation logic

See also: [Coding Standards](coding_standards.md), [Documentation Standards](../documentation_standards.md)
