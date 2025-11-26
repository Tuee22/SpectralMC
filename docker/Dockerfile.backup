# syntax=docker/dockerfile:1.4
# SpectralMC Dockerfile - GTX 970 Source Build
# Builds PyTorch and CuPy from source with sm_52 support for Maxwell GPUs

FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ENV PATH="/root/.local/bin:/usr/local/bin:$PATH"

# CuPy JIT compilation settings
ENV CUPY_COMPILE_WITH_PTX=1
ENV CUPY_NVCC_GENERATE_CODE=current

# Build environment for GTX 970 (compute capability 5.2)
ENV TORCH_CUDA_ARCH_LIST="5.2+PTX"
ENV USE_CUDA=1
ENV USE_CUDNN=1
ENV USE_NCCL=1

# 1) Update and install base tools
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
    && rm -rf /var/lib/apt/lists/*

# 2) Install Python 3.12 from deadsnakes
RUN add-apt-repository ppa:deadsnakes/ppa -y && \
    apt-get update && apt-get install -y --no-install-recommends \
    python3.12-venv \
    python3.12-dev \
    python3.12-lib2to3 \
    && rm -rf /var/lib/apt/lists/*

# 3) Set Python 3.12 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 && \
    update-alternatives --set python3 /usr/bin/python3.12 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1 && \
    update-alternatives --set python /usr/bin/python3.12

# 4) Install pip and upgrade tools
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12 && \
    python -m pip install --upgrade pip setuptools wheel

# 5) Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3.12 - && \
    poetry config virtualenvs.create false

# 6) Install Python build dependencies
RUN pip install --no-cache-dir setuptools wheel pyyaml typing_extensions numpy fastrlock

# 7) Work in /spectralmc
WORKDIR /spectralmc

# 8) Copy dependency files first (for better caching)
COPY pyproject.toml ./

# 9) Copy rest of repo contents
COPY ./ /spectralmc

# 10) Clone and build PyTorch from source
RUN echo "=== [1/6] Cloning PyTorch 2.7.0 ===" && \
    git clone --depth 1 --branch v2.7.0 --recursive \
        https://github.com/pytorch/pytorch.git /tmp/pytorch && \
    echo "✓ PyTorch cloned"

RUN echo "=== [2/6] Building PyTorch wheel ===" && \
    cd /tmp/pytorch && \
    pip install -r requirements.txt && \
    MAX_JOBS=$(nproc) python setup.py bdist_wheel || true && \
    ls -la /tmp/pytorch/dist/ && \
    ls /tmp/pytorch/dist/*.whl > /dev/null 2>&1 || (echo "FATAL: PyTorch wheel not produced" && exit 1) && \
    echo "✓ PyTorch wheel built"

RUN echo "=== [3/6] Installing PyTorch wheel ===" && \
    pip install /tmp/pytorch/dist/*.whl && \
    python -c "import torch; print(f'PyTorch {torch.__version__} installed')" && \
    echo "✓ PyTorch installed"

# 11) Clone and build CuPy from source
RUN echo "=== [4/6] Cloning CuPy 13.3.0 ===" && \
    pip install fastrlock && \
    git clone --depth 1 --branch v13.3.0 --recursive \
        https://github.com/cupy/cupy.git /tmp/cupy && \
    echo "✓ CuPy cloned"

RUN echo "=== [5/6] Building CuPy wheel ===" && \
    cd /tmp/cupy && \
    export CUPY_NVCC_GENERATE_CODE="arch=compute_52,code=sm_52" && \
    export CUPY_COMPILE_WITH_PTX=1 && \
    python setup.py bdist_wheel || true && \
    ls -la /tmp/cupy/dist/ && \
    ls /tmp/cupy/dist/*.whl > /dev/null 2>&1 || (echo "FATAL: CuPy wheel not produced" && exit 1) && \
    echo "✓ CuPy wheel built"

RUN echo "=== [6/6] Installing CuPy wheel ===" && \
    pip install /tmp/cupy/dist/*.whl && \
    python -c "import cupy; print(f'CuPy installed, CUDA {cupy.cuda.runtime.runtimeGetVersion()}')" && \
    echo "✓ CuPy installed"

# 12) Install SpectralMC dependencies and reinstall source builds
RUN echo "=== Installing SpectralMC dependencies ===" && \
    cd /spectralmc && \
    poetry install --with dev --no-interaction && \
    echo "=== Reinstalling source-built packages ===" && \
    pip install --force-reinstall --no-deps /tmp/pytorch/dist/*.whl && \
    pip install --force-reinstall --no-deps /tmp/cupy/dist/*.whl && \
    rm -rf /tmp/pytorch /tmp/cupy && \
    echo "=== Validating CUDA build ===" && \
    python -c "import torch; \
        cuda_ver = torch.version.cuda; \
        print(f'PyTorch CUDA version: {cuda_ver}'); \
        assert cuda_ver is not None, 'FATAL: PyTorch not compiled with CUDA'" && \
    python -c "import cupy; print(f'CuPy CUDA: {cupy.cuda.runtime.runtimeGetVersion()}')" && \
    echo "GTX970_SOURCE_BUILD=true" > /spectralmc/.build-marker && \
    python -c "import torch; print(f'TORCH_CUDA={torch.version.cuda}')" >> /spectralmc/.build-marker && \
    date >> /spectralmc/.build-marker && \
    echo "✓ Build validation passed"

# 13) Enforce CLAUDE.md testing policies: Block direct pytest and timeout commands
RUN if [ -f /usr/local/bin/pytest ]; then mv /usr/local/bin/pytest /usr/local/bin/pytest.real; fi && \
    echo '#!/bin/sh\necho "❌ ERROR: Direct pytest forbidden per CLAUDE.md"\necho "✅ USE: poetry run test-all [pytest-args]"\necho "📖 Examples:"\necho "  poetry run test-all              # Run all tests"\necho "  poetry run test-all -v           # Verbose output"\necho "  poetry run test-all -m gpu       # GPU tests only"\necho "📖 Reference: CLAUDE.md testing policies"\nexit 1' > /usr/local/bin/pytest && \
    chmod +x /usr/local/bin/pytest

RUN if [ -f /usr/bin/pytest ]; then mv /usr/bin/pytest /usr/bin/pytest.real; fi && \
    echo '#!/bin/sh\necho "❌ ERROR: Direct pytest forbidden per CLAUDE.md"\necho "✅ USE: poetry run test-all [pytest-args]"\necho "📖 Reference: CLAUDE.md testing policies"\nexit 1' > /usr/bin/pytest && \
    chmod +x /usr/bin/pytest

RUN if [ -f /usr/bin/timeout ]; then mv /usr/bin/timeout /usr/bin/timeout.real; fi && \
    echo '#!/bin/sh\nif echo "$*" | grep -qE "(pytest|test-|test_)"; then\n  echo "❌ ERROR: Timeout commands with tests forbidden per CLAUDE.md"\n  echo "✅ USE: poetry run test-all [pytest-args]"\n  echo "📖 Tests complete when they complete - patience required"\n  exit 1\nfi\nexec /usr/bin/timeout.real "$@"' > /usr/bin/timeout && \
    chmod +x /usr/bin/timeout

# 14) Expose ports
EXPOSE 6006 8888

# 15) Use tini as init
ENTRYPOINT ["/usr/bin/tini", "--"]

# 16) Default command
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", \
     "--NotebookApp.token=", "--NotebookApp.password=", "--notebook-dir=/spectralmc"]
