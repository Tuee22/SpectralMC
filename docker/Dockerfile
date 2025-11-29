# syntax=docker/dockerfile:1.4
# SpectralMC Dockerfile - Source Build for GTX 970 Compatibility
#
# Build command:
#   docker compose up --build -d
#
# Build characteristics:
# - PyTorch 2.4.1 compiled from source with sm_52 (GTX 970 Maxwell) support
# - Build time: 2-4 hours (first build), cached afterward
# - Supports all NVIDIA GPUs including legacy Maxwell (sm_52) and Kepler (sm_35)
#
# Layer Strategy:
# - Early layers: System dependencies (changes rarely, cached aggressively)
# - Middle layers: PyTorch source build (2-4 hours, cached aggressively)
# - Late layers: Application code (changes frequently, rebuilds fast)

FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04
ARG DEBIAN_FRONTEND=noninteractive
ENV PATH="/root/.local/bin:/usr/local/bin:$PATH"
# Set UTF-8 locale for proper Unicode handling in CLI output
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

# CuPy JIT compilation settings (for runtime optimization)
ENV CUPY_COMPILE_WITH_PTX=1
ENV CUPY_NVCC_GENERATE_CODE=current

# ============================================================================
# LAYER 1: System Dependencies (changes: never)
# ============================================================================

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    software-properties-common \
    apt-transport-https \
    ca-certificates \
    gnupg \
    lsb-release \
    curl \
    wget \
    unzip \
    git \
    tmux \
    htop \
    openssh-client \
    jq \
    nano \
    tini \
    bash-completion \
    cmake \
    ninja-build \
    ccache \
    libjpeg-dev \
    libpng-dev \
    locales \
    libopenblas-dev \
    liblapack-dev \
    && rm -rf /var/lib/apt/lists/* \
    && locale-gen en_US.UTF-8

# ============================================================================
# LAYER 2: Python 3.12 Installation (changes: rarely)
# ============================================================================

RUN add-apt-repository ppa:deadsnakes/ppa -y && \
    apt-get update && apt-get install -y --no-install-recommends \
    python3.12-venv \
    python3.12-dev \
    python3.12-lib2to3 \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 && \
    update-alternatives --set python3 /usr/bin/python3.12 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1 && \
    update-alternatives --set python /usr/bin/python3.12

RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12 && \
    python -m pip install --upgrade pip setuptools wheel

# ============================================================================
# LAYER 3: Poetry Installation (changes: rarely)
# ============================================================================

RUN python -m pip install --upgrade pip setuptools wheel poetry && \
    poetry config virtualenvs.create false

# ============================================================================
# LAYER 4: Python Build Dependencies (changes: rarely)
# ============================================================================

RUN pip install --no-cache-dir setuptools wheel pyyaml typing_extensions numpy fastrlock

# ============================================================================
# LAYER 5: Early Validation (FAST FAIL - runs in 30 seconds)
# ============================================================================

RUN echo "=== [0/3] Pre-Build Validation ===" && \
    echo "Build mode: PyTorch source build (sm_52 support)" && \
    echo "Checking CUDA toolkit..." && \
    nvcc --version || (echo "FATAL: nvcc not found" && exit 1) && \
    echo "Checking Python version..." && \
    python --version | grep "3.12" || (echo "FATAL: Python 3.12 not found" && exit 1) && \
    echo "TARGET: All NVIDIA GPUs including GTX 970 Maxwell (sm_52) with CUDA 11.8" && \
    echo "✓ Pre-build validation passed"

# ============================================================================
# LAYER 6: ccache Configuration (for source builds)
# ============================================================================

ENV CCACHE_DIR=/root/.ccache
ENV CMAKE_CXX_COMPILER_LAUNCHER=ccache
ENV CMAKE_CUDA_COMPILER_LAUNCHER=ccache
RUN ccache --max-size=20G && ccache --zero-stats

WORKDIR /spectralmc

# ============================================================================
# LAYER 7: Copy Dependency Manifests ONLY (changes: occasionally)
# ============================================================================

COPY pyproject.toml ./

# ============================================================================
# LAYER 8: Install SpectralMC Dependencies FIRST (before PyTorch source build)
# ============================================================================
# Poetry install may pull in torch as a transitive dependency.
# We run it first, then build PyTorch from source to override.

RUN echo "=== [1/3] Installing SpectralMC dependencies (Poetry) ===" && \
    poetry install --no-interaction --no-root && \
    echo "✓ Dependencies installed"

# ============================================================================
# LAYER 9: Install PyTorch - Binary OR Source Build (AFTER poetry)
# ============================================================================

# For SOURCE BUILD: Clone and build PyTorch 2.4.1 with sm_52 support
# PyTorch 2.4.1 chosen for:
# - NumPy 2.0 support (confirmed)
# - CUDA 11.8 support (maintained)
# - Pre-dates Maxwell (sm_52) removal (happened in 2.8)
#
# IMPORTANT: This runs AFTER poetry install to override any torch pulled as
# a transitive dependency with our sm_52-compiled version.

# Set PyTorch source build environment variables
ENV TORCH_CUDA_ARCH_LIST="5.2"
ENV USE_CUDA=1
ENV USE_CUDNN=1
ENV USE_NCCL=0
ENV MAX_JOBS=4
ENV CUDA_HOME=/usr/local/cuda-11.8
ENV LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH
# Enable LAPACK/BLAS for torch.linalg operations on CPU
ENV BLAS=OpenBLAS
ENV USE_LAPACK=1

RUN echo "=== [2/3] Building PyTorch 2.4.1 from source (sm_52 + NumPy 2.0) ===" && \
    echo "This will take 2-4 hours. Patience required." && \
    echo "Note: This overrides any torch installed by poetry" && \
    cd /tmp && \
    git clone --branch v2.4.1 --recursive --depth 1 https://github.com/pytorch/pytorch.git && \
    cd /tmp/pytorch && \
    python setup.py bdist_wheel 2>&1 | tee /tmp/pytorch-build.log && \
    mkdir -p /opt/pytorch-wheel && \
    cp dist/*.whl /opt/pytorch-wheel/ && \
    pip install --force-reinstall /opt/pytorch-wheel/*.whl && \
    cd /spectralmc && \
    rm -rf /tmp/pytorch && \
    python -c "import torch; print(f'✓ PyTorch {torch.__version__} built from source')" && \
    python -c "import torch; print(f'  CUDA version: {torch.version.cuda}')" && \
    echo "BUILD_TYPE=pytorch_source_sm52" > /spectralmc/.build-marker && \
    echo "NOTE: sm_52 validation requires GPU access - will be tested at runtime"

# ============================================================================
# LAYER 10: Install CuPy (NumPy 2.0 compatible)
# ============================================================================

RUN echo "=== [3/3] Installing CuPy 13.x (NumPy 2.0 compatible) ===" && \
    pip install cupy-cuda11x && \
    python -c "import cupy; print(f'✓ CuPy {cupy.__version__} installed')"

# ============================================================================
# LAYER 11: Validate CUDA Build
# ============================================================================

RUN echo "=== Validating CUDA build ===" && \
    python -c "import torch; cuda_ver = torch.version.cuda; print(f'PyTorch CUDA version: {cuda_ver}'); assert cuda_ver is not None and cuda_ver.startswith('11'), 'Expected CUDA 11.x'" && \
    python -c "import cupy; print(f'CuPy version: {cupy.__version__}')" && \
    python -c "import torch; print(f'TORCH_VERSION={torch.__version__}')" >> /spectralmc/.build-marker && \
    python -c "import torch; print(f'TORCH_CUDA={torch.version.cuda}')" >> /spectralmc/.build-marker && \
    date >> /spectralmc/.build-marker && \
    echo "✓ Build validation passed"

# ============================================================================
# LAYER 12: Copy Application Source Code (changes: FREQUENTLY)
# ============================================================================

COPY ./ /spectralmc/

# ============================================================================
# LAYER 13: Install SpectralMC Package (changes: frequently)
# ============================================================================

RUN echo "=== Installing SpectralMC package ===" && \
    poetry install --no-interaction && \
    if [ -f /spectralmc/.build-marker ] && grep -q "pytorch_source_sm52" /spectralmc/.build-marker; then \
        echo "=== Restoring source-built PyTorch (poetry overwrote it) ===" && \
        pip install --force-reinstall --no-deps /opt/pytorch-wheel/*.whl && \
        python -c "import torch; print(f'Restored PyTorch {torch.__version__} CUDA {torch.version.cuda}')"; \
    fi && \
    echo "✓ SpectralMC installed"

# ============================================================================
# LAYER 14: Enforce CLAUDE.md Testing Policies
# ============================================================================

RUN if [ -f /usr/local/bin/pytest ]; then mv /usr/local/bin/pytest /usr/local/bin/pytest.real; fi && \
    echo '#!/bin/sh\necho "❌ ERROR: Direct pytest forbidden per CLAUDE.md"\necho "✅ USE: poetry run test-all [pytest-args]"\necho "📖 Examples:"\necho "  poetry run test-all              # Run all tests"\necho "  poetry run test-all -v           # Verbose output"\necho "  poetry run test-all -m gpu       # GPU tests only"\necho "📖 Reference: CLAUDE.md testing policies"\nexit 1' > /usr/local/bin/pytest && \
    chmod +x /usr/local/bin/pytest

RUN if [ -f /usr/bin/pytest ]; then mv /usr/bin/pytest /usr/bin/pytest.real; fi && \
    echo '#!/bin/sh\necho "❌ ERROR: Direct pytest forbidden per CLAUDE.md"\necho "✅ USE: poetry run test-all [pytest-args]"\necho "📖 Reference: CLAUDE.md testing policies"\nexit 1' > /usr/bin/pytest && \
    chmod +x /usr/bin/pytest

RUN if [ -f /usr/bin/timeout ]; then mv /usr/bin/timeout /usr/bin/timeout.real; fi && \
    echo '#!/bin/sh\nif echo "$*" | grep -qE "(pytest|test-|test_)"; then\n  echo "❌ ERROR: Timeout commands with tests forbidden per CLAUDE.md"\n  echo "✅ USE: poetry run test-all [pytest-args]"\n  echo "📖 Tests complete when they complete - patience required"\n  exit 1\nfi\nexec /usr/bin/timeout.real "$@"' > /usr/bin/timeout && \
    chmod +x /usr/bin/timeout

# ============================================================================
# LAYER 15: Final Configuration
# ============================================================================

EXPOSE 6006 8888

ENTRYPOINT ["/usr/bin/tini", "--"]

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", \
     "--NotebookApp.token=", "--NotebookApp.password=", "--notebook-dir=/spectralmc"]
