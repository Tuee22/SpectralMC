# File: README.md
# Spectral Monte-Carlo Learning

**Status**: Reference only  
**Supersedes**: None  
**Referenced by**: AGENTS.md; CLAUDE.md  

> **Purpose**: Overview and quickstart for SpectralMC (install, usage, commands).
> **📖 Authoritative Reference**: [AGENTS.md](AGENTS.md)

## Overview

Spectral Monte-Carlo (SpectralMC) is an online machine-learning method that trains complex-valued neural networks (CVNNs)using Monte-Carlo data. It draws on techniques from Reinforcement Learning, particularly policy gradient methods. This method is especially useful when the desired quantity is an expected function of a simulated distribution.

While useful in multiple domains, SpectralMC is especially relevant to the field of Quantitative Finance, where stochastic processes are widely used. Traditional Monte-Carlo-based solutions can be compute-intensive, sometimes requiring large compute clusters. By contrast, SpectralMC trains and performs inference continuously on the GPU to reduce overall computational load. This allows efficient training and inference accuracy, with a much smaller computational footprint than traditional Monte-Carlo methods.

If you want to read the entire codebase via a single end-to-end path, start with the GPU-only
test `tests/test_e2e/test_full_stack_cvnn_pricer.py`. It walks through CVNN construction,
GBM simulation setup, training, snapshotting, blockchain commit/reload, and deterministic
inference—touching the major subsystems in one place.

## Key Features

### Training Lifecycle Overview

SpectralMC's complete training workflow from simulation to production deployment:

```mermaid
flowchart TB
  SimParams[1. Define Simulation Parameters - BlackScholesConfig]
  SobolSample[2. Sobol Sampler - Generate Quasi-Random Contracts]
  MCSimulation[3. GPU Monte Carlo - Price Contracts with GBM]
  FFT[4. FFT - Estimate Characteristic Function]
  CVNNTrain[5. CVNN Training - Approximate Spectrum]
  BlockchainCommit[6. Blockchain Commit - Version Model]
  Inference[7. Inference - Production Pricing]

  SimParams --> SobolSample
  SobolSample --> MCSimulation
  MCSimulation --> FFT
  FFT --> CVNNTrain
  CVNNTrain --> BlockchainCommit
  BlockchainCommit --> Inference

  CVNNTrain -->|Next Batch| SobolSample
```

### Component Architecture

SpectralMC's tightly-coupled components in the GbmCVNNPricer:

```mermaid
flowchart TB
  GbmCVNNPricer[GbmCVNNPricer - Training Orchestrator]
  SobolSampler[SobolSampler - Quasi-Random Contract Generation]
  GBMSimulator[GBM Simulator - Numba CUDA Kernel]
  FFTEngine[FFT Engine - CuPy cuFFT]
  CVNNModel[CVNN Model - PyTorch Complex Network]
  Optimizer[Adam Optimizer - Complex Gradients]

  GbmCVNNPricer --> SobolSampler
  GbmCVNNPricer --> GBMSimulator
  GbmCVNNPricer --> FFTEngine
  GbmCVNNPricer --> CVNNModel
  GbmCVNNPricer --> Optimizer

  SobolSampler -->|Contract Parameters| GBMSimulator
  GBMSimulator -->|MC Samples| FFTEngine
  FFTEngine -->|Characteristic Function| CVNNModel
  CVNNModel -->|Predictions| Optimizer
  Optimizer -->|Updated Weights| CVNNModel
```

### GPU Execution Pipeline

Pure-GPU workflow with zero CPU transfers during training:

```mermaid
flowchart TB
  GPUStart[GPU Memory - Contract Parameters]
  NumbaKernel[Numba CUDA - GBM Simulation]
  DLPackTransfer[DLPack Zero-Copy Transfer]
  CuPyFFT[CuPy - FFT Computation]
  PyTorchCVNN[PyTorch - CVNN Forward/Backward]
  GPUEnd[GPU Memory - Updated Model Weights]

  GPUStart --> NumbaKernel
  NumbaKernel --> DLPackTransfer
  DLPackTransfer --> CuPyFFT
  CuPyFFT --> PyTorchCVNN
  PyTorchCVNN --> GPUEnd
  GPUEnd -->|Next Batch| GPUStart
```

### Core Capabilities

- **Monte-Carlo Simulation**: Generates a finite sample from a parametric distribution directly on the GPU.
- **Fourier Transform**: Uses Fast Fourier Transform (FFT) to estimate the sample's characteristic function.
- **CVNN Training**: Updates the parameters of a complex-valued neural network to approximate the characteristic function.
- **CVNN Inference**: Produces an estimated distribution (via the characteristic function), enabling computation of means, moments, quantiles, and other metrics.
- **Blockchain Model Versioning**: Production-ready S3-based version control for trained models with:
  - Automatic commits during training (explicit commit plans for final/periodic checkpoints)
  - Immutable version history with SHA256 content addressing
  - Atomic commits with CAS (Compare-And-Swap) using ETag
  - InferenceClient for production model serving (pinned/tracking modes)
  - Chain integrity verification to detect tampering
  - Automated garbage collection for old versions
  - TensorBoard integration for metrics logging
  - CLI tools for version management (83% test coverage)

## Getting Started

## Prerequisites

- A ubuntu 24.04 machine with an NVIDIA GPU


## Installation

1. Use a Ubuntu 24.04 machine with an nvidia GPU. If you're on AWS, for example, you might choose a g5.xlarge instance, running Ubuntu 24.04.

2. Clone this repository:
```bash
# File: README.md
git clone https://github.com/Tuee22/SpectralMC.git
cd SpectralMC
```

3. Install Docker, the NVIDIA drivers, and the NVIDIA Container Toolkit (follow NVIDIA's installation docs). After installation, reboot to ensure the driver is active and verify Docker can access the GPU:
```bash
# File: README.md
docker run --gpus all --rm nvidia/cuda:12.4.1-base-ubuntu24.04 nvidia-smi
```

4. To interact with the container manually:

```bash
# File: README.md
cd docker
docker compose up -d
docker compose exec -it spectralmc bash
```

This will open a bash terminal, which will allow you to explicitly call the python library, eg

```bash
# File: README.md
python -m spectralmc.gbm
```


## Usage Examples

### Automatic Training Commits

Train models with automatic blockchain commits:

```python
# File: README.md
import asyncio
import torch
from spectralmc.gbm_trainer import (
    FinalAndIntervalCommit,
    FinalCommit,
    GbmCVNNPricer,
    TrainingConfig,
    build_training_config,
)
from spectralmc.result import Failure, Success
from spectralmc.storage import AsyncBlockchainModelStore
from spectralmc.testing import make_gbm_cvnn_config

assert torch.cuda.is_available(), "CUDA required for training"
config = make_gbm_cvnn_config(torch.nn.Linear(5, 100).to("cuda"))

# Auto-commit only works when train() is called from a synchronous context
store = asyncio.run(AsyncBlockchainModelStore("my-model-bucket").__aenter__())
try:
    match GbmCVNNPricer.create(config):
        case Success(pricer):
            training_config = build_training_config(
                num_batches=1000,
                batch_size=32,
                learning_rate=0.001,
            ).unwrap()

            # Auto-commit after training completes
            pricer.train(
                training_config,
                blockchain_store=store,
                commit_plan=FinalCommit(
                    commit_message_template="Final: step={step}, loss={loss:.4f}"
                ),
            )

            # Or commit every 100 batches during training
            pricer.train(
                training_config,
                blockchain_store=store,
                commit_plan=FinalAndIntervalCommit(
                    interval=100, commit_message_template="Checkpoint step={step}"
                ),
            )
        case Failure(error):
            raise RuntimeError(f"Failed to create pricer: {error}")
finally:
    asyncio.run(store.__aexit__(None, None, None))
```

### Manual Model Commits

```python
# File: README.md
import torch
from spectralmc.runtime import get_torch_handle
from spectralmc.gbm_trainer import GbmCVNNPricer
from spectralmc.storage import AsyncBlockchainModelStore, commit_snapshot
from spectralmc.testing import make_gbm_cvnn_config
from spectralmc.result import Failure, Success

get_torch_handle()
model = torch.nn.Linear(5, 5).to("cuda")
config = make_gbm_cvnn_config(model)

match GbmCVNNPricer.create(config):
    case Failure(error):
        raise RuntimeError(f"Failed to create pricer: {error}")
    case Success(pricer):
        match pricer.snapshot():
            case Failure(error):
                raise RuntimeError(f"Failed to snapshot pricer: {error}")
            case Success(snapshot):
                pass  # Deterministic GbmCVNNPricerConfig

# Commit to blockchain storage
async with AsyncBlockchainModelStore("my-model-bucket") as store:
    version = await commit_snapshot(
        store,
        snapshot,
        message="Trained for 1000 epochs, loss=0.001",
    )
    print(f"Committed version {version.counter}")
```

### Production Inference

```python
# File: README.md
import torch
from spectralmc.storage import (
    AsyncBlockchainModelStore,
    InferenceClient,
    TrackingMode,
    pinned_mode,
)
from spectralmc.result import Success
from spectralmc.testing import make_gbm_cvnn_config

model_template = torch.nn.Linear(5, 100).to("cuda")
config_template = make_gbm_cvnn_config(model_template)

async with AsyncBlockchainModelStore("my-model-bucket") as store:
    # Pinned mode: Always serve version 42 (production stability)
    match pinned_mode(42):
        case Success(mode):
            async with InferenceClient(
                mode=mode,
                poll_interval=60.0,
                store=store,
                model_template=model_template,
                config_template=config_template,
            ) as client:
                snapshot = client.get_model()
                # Run inference with snapshot.cvnn

    # Tracking mode: follow latest version automatically
    async with InferenceClient(
        mode=TrackingMode(),
        poll_interval=30.0,
        store=store,
        model_template=model_template,
        config_template=config_template,
    ) as client:
        latest_snapshot = client.get_model()
        # Run inference with latest_snapshot.cvnn
```

### CLI Tools

```bash
# File: README.md
# List all versions
python -m spectralmc.storage list-versions my-model-bucket

# Verify chain integrity
python -m spectralmc.storage verify my-model-bucket

# Garbage collection (keep last 10 versions)
python -m spectralmc.storage gc-run my-model-bucket 10 --yes

# Log to TensorBoard
python -m spectralmc.storage tensorboard-log my-model-bucket
tensorboard --logdir=runs/blockchain_models
```

For complete documentation, see [CLAUDE.md](CLAUDE.md).

## Testing

SpectralMC enforces poetry-based test execution to ensure consistency across environments. All tests must run via poetry commands inside the Docker container.

### Running Tests

**All commands must run inside the Docker container:**

```bash
# File: README.md
# Start the container
cd docker && docker compose up -d

# Run all tests (CPU + GPU)
docker compose -f docker/docker-compose.yml exec spectralmc poetry run test-all

# Run with verbose output
docker compose -f docker/docker-compose.yml exec spectralmc poetry run test-all -v

# Run specific test file
docker compose -f docker/docker-compose.yml exec spectralmc poetry run test-all tests/test_gbm.py

# Run specific test function
docker compose -f docker/docker-compose.yml exec spectralmc poetry run test-all tests/test_gbm.py::test_gbm_simulation

# Run tests matching keyword
docker compose -f docker/docker-compose.yml exec spectralmc poetry run test-all -k "sobol"

# Run with coverage report
docker compose -f docker/docker-compose.yml exec spectralmc poetry run test-all --cov=spectralmc --cov-report=term-missing

# Stop on first failure
docker compose -f docker/docker-compose.yml exec spectralmc poetry run test-all -x
```

### Important Testing Rules

**REQUIRED by CLAUDE.md:**
- ❌ **NEVER** run `pytest` directly (blocked by Dockerfile)
- ❌ **NEVER** use timeout commands with tests (blocked by Dockerfile)
- ✅ **ALWAYS** use `poetry run test-all` (with or without arguments)
- ✅ **ALWAYS** redirect test output to files for complete analysis:
  ```bash
  # File: README.md
  docker compose -f docker/docker-compose.yml exec spectralmc poetry run test-all > /tmp/test-output.txt 2>&1
  ```

### Test Organization

- **Test directory**: `tests/`
- **GPU required**: All tests require GPU - missing GPU causes test failure
- **No fallbacks**: Silent CPU fallbacks are forbidden
- **Fixtures**: Global GPU memory cleanup in `tests/conftest.py`

### Type Checking

Run mypy from repository root (no path arguments):

```bash
# File: README.md
docker compose -f docker/docker-compose.yml exec spectralmc mypy
```

This checks all configured paths (`src/spectralmc`, `tests`, `examples`) per the appropriate pyproject file (pyproject.binary.toml or pyproject.source.toml - they share identical `[tool.mypy]` configuration).

## Contribution

We welcome contributions to SpectralMC. A few steps: 

- Fork this repo.
- Create a new branch for your new feature or bugfix.
- Open a pull request describing changes.

## License

This project is distributed under the MIT License. See the LICENSE file for details.


## Contact

Author: matt@resolvefintech.com


Send any questions or suggestions to this email address.
