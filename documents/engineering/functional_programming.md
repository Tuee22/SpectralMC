# Functional Programming in SpectralMC

## Overview

SpectralMC adopts **pure functional programming patterns** for error handling, state management, and data transformations. This document defines the functional programming standards that all SpectralMC code must follow.

**Core Principles**:
- **No legacy APIs**: All code must use pure functional patterns (no imperative/OOP error handling)
- **Algebraic Data Types (ADTs)**: Model errors and domain states as frozen dataclasses
- **Result Types**: Return `Result[T, E]` instead of raising exceptions for expected errors
- **Pattern Matching**: Use Python 3.10+ `match/case` for exhaustive error handling
- **Immutability**: All data structures are immutable by default (see [Immutability Doctrine](./immutability_doctrine.md))

## Related Documentation

- [Coding Standards](./coding_standards.md) - Type safety, formatting, and stubs
- [Immutability Doctrine](./immutability_doctrine.md) - Immutable data structures
- [Pydantic Patterns](./pydantic_patterns.md) - Model validation and serialization

## Algebraic Data Types (ADTs)

### What Are ADTs?

ADTs represent **sum types** - types that can be one of several variants. In SpectralMC, we use frozen dataclasses to model error types, operation results, and domain states.

**Key Properties**:
- `@dataclass(frozen=True)` - Immutable by default
- Explicit variants with distinct fields
- Type-safe pattern matching with mypy
- Self-documenting error taxonomy

### Error ADT Pattern

**Primary Example: S3 Operations**

```python
from dataclasses import dataclass
from typing import Literal

@dataclass(frozen=True)
class S3BucketNotFound:
    """S3 bucket does not exist."""
    kind: Literal["S3BucketNotFound"] = "S3BucketNotFound"
    bucket: str
    original_error: Exception

@dataclass(frozen=True)
class S3ObjectNotFound:
    """S3 object does not exist at the given key."""
    kind: Literal["S3ObjectNotFound"] = "S3ObjectNotFound"
    bucket: str
    key: str
    original_error: Exception

@dataclass(frozen=True)
class S3AccessDenied:
    """Insufficient permissions to access S3 resource."""
    kind: Literal["S3AccessDenied"] = "S3AccessDenied"
    bucket: str
    key: str | None
    original_error: Exception

@dataclass(frozen=True)
class S3NetworkError:
    """Network-level error (timeout, connection failed)."""
    kind: Literal["S3NetworkError"] = "S3NetworkError"
    bucket: str
    key: str | None
    original_error: Exception

@dataclass(frozen=True)
class S3UnknownError:
    """Unexpected S3 error not covered by other variants."""
    kind: Literal["S3UnknownError"] = "S3UnknownError"
    bucket: str
    key: str | None
    original_error: Exception

# Sum type: S3OperationError is ONE OF these variants
S3OperationError = (
    S3BucketNotFound
    | S3ObjectNotFound
    | S3AccessDenied
    | S3NetworkError
    | S3UnknownError
)
```

**Why This Pattern?**

1. **Exhaustive Error Handling**: Pattern matching forces you to handle ALL error cases
2. **Type Safety**: mypy verifies you handle every variant
3. **Self-Documenting**: Error taxonomy is explicit in the type definition
4. **No Silent Failures**: Cannot ignore errors without explicit handling
5. **Preserves Context**: Each variant captures relevant diagnostic information

### Domain State ADTs

Beyond errors, use ADTs to model domain states:

```python
@dataclass(frozen=True)
class TrainingInProgress:
    kind: Literal["TrainingInProgress"] = "TrainingInProgress"
    current_epoch: int
    current_loss: float

@dataclass(frozen=True)
class TrainingConverged:
    kind: Literal["TrainingConverged"] = "TrainingConverged"
    final_epoch: int
    final_loss: float

@dataclass(frozen=True)
class TrainingDiverged:
    kind: Literal["TrainingDiverged"] = "TrainingDiverged"
    diverged_at_epoch: int
    loss_value: float

TrainingState = TrainingInProgress | TrainingConverged | TrainingDiverged
```

## Result Types

### The Result[T, E] Pattern

The `Result[T, E]` type represents operations that can succeed with value `T` or fail with error `E`.

**Definition**:

```python
from dataclasses import dataclass
from typing import TypeVar, Generic

T = TypeVar("T")
E = TypeVar("E")

@dataclass(frozen=True)
class Success(Generic[T]):
    """Operation succeeded with value."""
    value: T

@dataclass(frozen=True)
class Failure(Generic[E]):
    """Operation failed with error."""
    error: E

# Result is either Success OR Failure
type Result[T, E] = Success[T] | Failure[E]
```

### When to Use Result Types

**Use Result[T, E] for**:
- ✅ Expected errors (file not found, network timeout, validation failure)
- ✅ Operations that can fail for known reasons
- ✅ External I/O (S3, database, network)
- ✅ User input validation

**Use exceptions for**:
- ✅ Programming errors (assertion failures, type errors)
- ✅ Unrecoverable errors (out of memory, system errors)
- ✅ Internal invariant violations

### Result Type Examples

**S3 Get Object**:

```python
async def get_object_safe(
    client: S3Client, bucket: str, key: str
) -> Result[bytes, S3OperationError]:
    """Fetch S3 object, returning Result instead of raising exceptions."""
    try:
        response = await client.get_object(Bucket=bucket, Key=key)
        body = await response["Body"].read()
        return Success(body)
    except ClientError as e:
        error_code = e.response["Error"]["Code"]
        if error_code == "NoSuchBucket":
            return Failure(S3BucketNotFound(bucket=bucket, original_error=e))
        elif error_code == "NoSuchKey":
            return Failure(S3ObjectNotFound(bucket=bucket, key=key, original_error=e))
        elif error_code == "AccessDenied":
            return Failure(S3AccessDenied(bucket=bucket, key=key, original_error=e))
        else:
            return Failure(S3UnknownError(bucket=bucket, key=key, original_error=e))
    except (BotocoreError, aiohttp.ClientError) as e:
        return Failure(S3NetworkError(bucket=bucket, key=key, original_error=e))
```

**Model Loading**:

```python
async def load_model_version(
    store: AsyncBlockchainModelStore, version_counter: int
) -> Result[ModelSnapshot, LoadError]:
    """Load model version, returning Result for expected errors."""
    # Get version metadata
    versions_result = await store.list_all_versions()
    match versions_result:
        case Failure(error):
            return Failure(LoadError.from_s3_error(error))
        case Success(versions):
            pass

    # Find requested version
    target_version = next(
        (v for v in versions if v.counter == version_counter), None
    )
    if target_version is None:
        return Failure(
            VersionNotFound(counter=version_counter, available=versions)
        )

    # Load checkpoint
    checkpoint_result = await store.get_checkpoint(target_version)
    match checkpoint_result:
        case Failure(error):
            return Failure(LoadError.from_s3_error(error))
        case Success(checkpoint_bytes):
            snapshot = deserialize_checkpoint(checkpoint_bytes)
            return Success(snapshot)
```

## Pattern Matching

### Exhaustive Error Handling

Python 3.10+ `match/case` enables type-safe, exhaustive error handling.

**Basic Pattern**:

```python
from typing import assert_never

result = await get_object_safe(client, bucket, key)

match result:
    case Success(data):
        # Handle success case
        process_data(data)
    case Failure(error):
        # Pattern match on error variants
        match error:
            case S3BucketNotFound(bucket=bucket_name):
                logger.error(f"Bucket does not exist: {bucket_name}")
                # Handle missing bucket
            case S3ObjectNotFound(bucket=b, key=k):
                logger.error(f"Object not found: s3://{b}/{k}")
                # Handle missing object
            case S3AccessDenied(bucket=b, key=k):
                logger.error(f"Access denied: s3://{b}/{k}")
                # Handle permission error
            case S3NetworkError(original_error=e):
                logger.error(f"Network error: {e}")
                # Retry or fail
            case S3UnknownError(original_error=e):
                logger.error(f"Unknown S3 error: {e}")
                # Escalate to monitoring
            case _:
                # Exhaustiveness check - will fail type checking if new variant added
                assert_never(error)
```

### Why assert_never()?

The `assert_never()` function provides **compile-time exhaustiveness checking**:

```python
from typing import Never

def assert_never(value: Never) -> Never:
    """Type-safe exhaustiveness check for pattern matching."""
    raise AssertionError(f"Unhandled value: {value} ({type(value).__name__})")
```

**What it does**:
1. **Compile-time**: mypy verifies all cases are handled (the `case _:` branch is unreachable)
2. **Runtime safety**: If somehow a new variant appears at runtime, raises AssertionError
3. **Refactoring safety**: Adding new ADT variant causes type error in all match statements

**Example**:

```python
# If we add a new variant to S3OperationError:
@dataclass(frozen=True)
class S3RateLimited:
    kind: Literal["S3RateLimited"] = "S3RateLimited"
    bucket: str
    retry_after: int

S3OperationError = (
    S3BucketNotFound
    | S3ObjectNotFound
    | S3AccessDenied
    | S3NetworkError
    | S3UnknownError
    | S3RateLimited  # New variant
)

# ALL existing match statements will fail type checking:
# error: Argument 1 to "assert_never" has incompatible type "S3RateLimited"; expected "Never"
```

### Nested Pattern Matching

Combine Result and ADT pattern matching:

```python
async def verify_chain(store: AsyncBlockchainModelStore) -> Result[None, VerifyError]:
    """Verify blockchain integrity."""
    versions_result = await store.list_all_versions()

    match versions_result:
        case Success(versions):
            # Continue with verification
            if not versions:
                return Failure(EmptyChain())

            # Check genesis block
            genesis = versions[0]
            if genesis.counter != 0:
                return Failure(InvalidGenesis(version=genesis))

            # Check chain linking
            for i in range(1, len(versions)):
                prev = versions[i - 1]
                curr = versions[i]
                if curr.parent_hash != prev.content_hash:
                    return Failure(BrokenChain(prev=prev, curr=curr))

            return Success(None)

        case Failure(error):
            # Map S3 errors to verification errors
            match error:
                case S3BucketNotFound(bucket=b):
                    return Failure(ChainNotFound(bucket=b))
                case S3NetworkError():
                    return Failure(VerificationNetworkError(original=error))
                case _:
                    return Failure(VerificationS3Error(original=error))
```

## No Legacy APIs Policy

**CRITICAL POLICY**: SpectralMC does not maintain backward compatibility with imperative/OOP error handling.

### Prohibited Patterns

**❌ NEVER use these patterns**:

```python
# ❌ Exception-based error handling for expected errors
def load_model(version: int) -> ModelSnapshot:
    if version not in available_versions:
        raise VersionNotFoundError(f"Version {version} not found")
    return load_from_disk(version)

# ❌ Returning None for errors
def get_object(bucket: str, key: str) -> bytes | None:
    try:
        return s3_client.get_object(Bucket=bucket, Key=key)
    except ClientError:
        return None  # Lost error context!

# ❌ Returning error codes
def verify_chain(store: Store) -> tuple[bool, str]:
    if not store.exists():
        return False, "Chain not found"
    return True, ""

# ❌ Mutable error state
class Operation:
    def __init__(self):
        self.error: str | None = None
        self.success: bool = False

    def run(self):
        try:
            self.do_work()
            self.success = True
        except Exception as e:
            self.error = str(e)
            self.success = False
```

### Required Patterns

**✅ ALWAYS use these patterns**:

```python
# ✅ Result type for expected errors
def load_model(version: int) -> Result[ModelSnapshot, LoadError]:
    if version not in available_versions:
        return Failure(VersionNotFound(version=version, available=available_versions))
    return Success(load_from_disk(version))

# ✅ Result type preserves error context
async def get_object(bucket: str, key: str) -> Result[bytes, S3OperationError]:
    try:
        response = await s3_client.get_object(Bucket=bucket, Key=key)
        return Success(await response["Body"].read())
    except ClientError as e:
        return Failure(classify_s3_error(e, bucket, key))

# ✅ Result type for verification
async def verify_chain(store: Store) -> Result[None, VerifyError]:
    versions_result = await store.list_all_versions()
    match versions_result:
        case Success(versions):
            if not versions:
                return Failure(EmptyChain())
            # ... verification logic
            return Success(None)
        case Failure(error):
            return Failure(VerificationError.from_s3_error(error))

# ✅ Immutable error state
@dataclass(frozen=True)
class OperationResult:
    snapshot: ModelSnapshot | None
    error: LoadError | None
```

### Migration Strategy

**NO gradual migration** - functional patterns are adopted immediately:

1. **New code**: MUST use Result types and ADTs from day one
2. **Refactoring**: Replace entire modules atomically (no hybrid states)
3. **Breaking changes**: Accepted and expected - update all callers simultaneously
4. **No deprecation warnings**: Old APIs are deleted, not deprecated

**Example Migration**:

```python
# BEFORE: Imperative exception-based code (DELETE THIS)
class ModelStore:
    def get_version(self, counter: int) -> ModelVersion:
        """Raises VersionNotFoundError if version doesn't exist."""
        try:
            return self._load_version(counter)
        except FileNotFoundError:
            raise VersionNotFoundError(f"Version {counter} not found")

# AFTER: Functional Result-based code (REPLACE WITH THIS)
class ModelStore:
    async def get_version(
        self, counter: int
    ) -> Result[ModelVersion, LoadError]:
        """Returns Result - Success with version or Failure with error."""
        versions_result = await self.list_all_versions()
        match versions_result:
            case Success(versions):
                version = next((v for v in versions if v.counter == counter), None)
                if version is None:
                    return Failure(VersionNotFound(counter=counter))
                return Success(version)
            case Failure(error):
                return Failure(LoadError.from_s3_error(error))
```

## Error Handling Best Practices

### The _raise() Helper

For cases where you need to convert Result to exception (e.g., at system boundaries):

```python
from typing import NoReturn

def _raise(error: E) -> NoReturn:
    """Convert error ADT to exception and raise.

    Use ONLY at system boundaries where exceptions are required.
    Internal code should propagate Result types.
    """
    raise RuntimeError(f"Operation failed: {error}")

# Usage: CLI commands (system boundary)
async def cmd_verify(bucket: str) -> int:
    """CLI command - uses exceptions at boundary."""
    async with AsyncBlockchainModelStore(bucket) as store:
        result = await verify_chain(store)
        match result:
            case Success(_):
                print("✓ Chain integrity verified")
                return 0
            case Failure(error):
                match error:
                    case EmptyChain():
                        print("✗ Chain is empty")
                        return 1
                    case InvalidGenesis(version=v):
                        print(f"✗ Invalid genesis block: {v}")
                        return 1
                    case BrokenChain(prev=prev, curr=curr):
                        print(f"✗ Broken chain link: {prev.counter} -> {curr.counter}")
                        return 1
                    case _:
                        # Unknown error - convert to exception at boundary
                        _raise(error)
```

**When to use _raise()**:
- ✅ CLI command error handling (exit codes + messages)
- ✅ HTTP API endpoints (convert to HTTP error responses)
- ✅ Test assertions (convert Result to exception for pytest)

**When NOT to use _raise()**:
- ❌ Internal library code (propagate Result types)
- ❌ Service layer logic (use pattern matching)
- ❌ Data processing pipelines (chain Results with monadic operations)

### Error Transformation

Transform errors between layers:

```python
@dataclass(frozen=True)
class VerificationError:
    """High-level verification error."""
    kind: str
    message: str
    underlying: S3OperationError | None

    @staticmethod
    def from_s3_error(error: S3OperationError) -> "VerificationError":
        """Transform S3 error to verification error."""
        match error:
            case S3BucketNotFound(bucket=b):
                return VerificationError(
                    kind="ChainNotFound",
                    message=f"Blockchain not found in bucket: {b}",
                    underlying=error,
                )
            case S3NetworkError():
                return VerificationError(
                    kind="NetworkError",
                    message="Network error during verification",
                    underlying=error,
                )
            case _:
                return VerificationError(
                    kind="UnknownError",
                    message=f"Unknown error: {error}",
                    underlying=error,
                )
```

## Testing Functional Code

### Testing Result Types

```python
async def test_get_object_success():
    """Test successful S3 get returns Success."""
    result = await get_object_safe(mock_client, "my-bucket", "my-key")
    match result:
        case Success(data):
            assert data == b"expected content"
        case Failure(error):
            pytest.fail(f"Expected Success, got Failure: {error}")

async def test_get_object_not_found():
    """Test missing object returns Failure with S3ObjectNotFound."""
    result = await get_object_safe(mock_client, "my-bucket", "missing-key")
    match result:
        case Success(_):
            pytest.fail("Expected Failure, got Success")
        case Failure(error):
            match error:
                case S3ObjectNotFound(bucket=b, key=k):
                    assert b == "my-bucket"
                    assert k == "missing-key"
                case _:
                    pytest.fail(f"Expected S3ObjectNotFound, got {type(error).__name__}")
```

### Testing Exhaustiveness

Verify pattern matching is exhaustive:

```python
def test_pattern_matching_exhaustiveness():
    """Verify all S3OperationError variants are handled."""
    # This test should fail type checking if new variant added
    def handle_error(error: S3OperationError) -> str:
        match error:
            case S3BucketNotFound():
                return "bucket"
            case S3ObjectNotFound():
                return "object"
            case S3AccessDenied():
                return "access"
            case S3NetworkError():
                return "network"
            case S3UnknownError():
                return "unknown"
            case _:
                assert_never(error)  # Type error if variant missing

    # Test all variants
    assert handle_error(S3BucketNotFound("b", Exception())) == "bucket"
    assert handle_error(S3ObjectNotFound("b", "k", Exception())) == "object"
    # ... etc
```

## Summary

**Functional Programming Checklist**:

- [ ] Use `@dataclass(frozen=True)` for all ADTs
- [ ] Model errors as sum types with explicit variants
- [ ] Return `Result[T, E]` for expected errors
- [ ] Use pattern matching with `match/case` for exhaustive handling
- [ ] Include `assert_never()` in `case _:` branches
- [ ] Never use exceptions for expected errors
- [ ] Never return `None` to indicate errors
- [ ] Never use mutable error state
- [ ] Transform errors between layers explicitly
- [ ] Use `_raise()` ONLY at system boundaries
- [ ] Test all Result variants
- [ ] Verify exhaustiveness in tests

**Related Documents**:
- [Coding Standards](./coding_standards.md) - Type safety enforcement
- [Immutability Doctrine](./immutability_doctrine.md) - Immutable data structures
- [Pydantic Patterns](./pydantic_patterns.md) - Model validation
