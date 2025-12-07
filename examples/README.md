# File: examples/README.md
# SpectralMC Examples

**Status**: Reference only  
**Supersedes**: None  
**Referenced by**: README.md  

> **Purpose**: Describe available SpectralMC example scripts and how to run them.
> **📖 Authoritative Reference**: [AGENTS.md](../AGENTS.md)

This directory contains example scripts demonstrating blockchain model storage functionality.

## Available Examples

### 1. blockchain_storage_basic.py
Basic blockchain storage operations:
- Creating a blockchain model store
- Committing checkpoints
- Retrieving versions by ID
- Loading checkpoint data
- Verifying chain integrity

**Run**: `python examples/blockchain_storage_basic.py`

### 2. blockchain_integrity_check.py
Integrity verification and tamper detection:
- Hash computation and determinism
- Chain linking verification
- Tamper detection demonstration
- Version immutability verification

**Run**: `python examples/blockchain_integrity_check.py`

## Note on Serialization

The serialization examples have been removed as they require deep understanding of the internal Pydantic model structure. For serialization examples, please refer to the test suite in `tests/test_serialization/`.

## Running Examples

All examples should be run from the project root inside the Docker container:

```bash
# File: examples/README.md
docker compose -f docker/docker-compose.yml exec spectralmc python examples/blockchain_storage_basic.py
docker compose -f docker/docker-compose.yml exec spectralmc python examples/blockchain_integrity_check.py
```

## Expected Output

All examples print step-by-step progress and should complete with a success message (✓).
