# syntax=docker/dockerfile:1.4
# SpectralMC Dockerfile - Binary Build (Fast)
#
# This Dockerfile is for BINARY builds using pre-compiled PyTorch wheels.
# For SOURCE builds (GTX 970 support), use Dockerfile.source
#
# Build command:
#   docker compose up --build -d
#
# Build characteristics:
# - PyTorch 2.7.1 binary wheel (CUDA 12.8)
# - Build time: 10-15 minutes
# - GPU support: Compute capability 6.0+ (GTX 1060+, RTX series, including RTX 5090)
#
# For legacy GPU support (GTX 970, compute capability 5.2):
#   BUILD_FROM_SOURCE=true docker compose up --build -d

FROM nvidia/cuda:12.8.1-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG BUILD_FROM_SOURCE=false

# Environment configuration (single source of truth)
ENV PATH="/root/.local/bin:/usr/local/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH="/spectralmc" \
    PYTHONPYCACHEPREFIX="/opt/pycache" \
    CUPY_COMPILE_WITH_PTX=1 \
    CUPY_NVCC_GENERATE_CODE=current \
    POETRY_NO_INTERACTION=1 \
    POETRY_HOME=/usr/local \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    XDG_CACHE_HOME="/opt/spectralmc/cache" \
    MYPY_CACHE_DIR=/opt/spectralmc/mypy_cache \
    RUFF_CACHE_DIR=/opt/spectralmc/ruff_cache \
    PYTEST_CACHE_DIR=/opt/spectralmc/pytest_cache \
    TLA_JAR=/opt/tla/tla2tools.jar \
    DOCKER_CONTAINER=1 \
    CI=false

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
    ca-certificates \
    jq \
    gnupg \
    nano \
    tini \
    bash-completion \
    openjdk-17-jre-headless \
    && rm -rf /var/lib/apt/lists/*

# 1.25) Install TLA+ tools (TLC)
RUN TLA_VERSION=1.7.3 && \
    mkdir -p /opt/tla && \
    wget -q https://github.com/tlaplus/tlaplus/releases/download/v${TLA_VERSION}/tla2tools.jar \
        -O /opt/tla/tla2tools.jar

# 1.5) Install protoc from GitHub (need >= 3.19 for protobuf 6.x compatibility)
RUN PROTOC_VERSION=29.5 && \
    wget -q https://github.com/protocolbuffers/protobuf/releases/download/v${PROTOC_VERSION}/protoc-${PROTOC_VERSION}-linux-x86_64.zip && \
    unzip -q protoc-${PROTOC_VERSION}-linux-x86_64.zip -d /usr/local && \
    rm protoc-${PROTOC_VERSION}-linux-x86_64.zip && \
    protoc --version

# 2) Install Python 3.12 from deadsnakes
RUN add-apt-repository ppa:deadsnakes/ppa -y && \
    apt-get update && apt-get install -y --no-install-recommends \
    python3.12-venv \
    python3.12-dev \
    python3.12-lib2to3 \
    && rm -rf /var/lib/apt/lists/*

# 4) Set Python 3.12 as default for python3 and python
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 && \
    update-alternatives --set python3 /usr/bin/python3.12 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1 && \
    update-alternatives --set python /usr/bin/python3.12

# 5) Install pip for Python 3.12, then install Poetry via pip (single command)
# Poetry virtual environment config is handled by poetry.toml (virtualenvs.create = false)
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12 && \
    python -m pip install --upgrade pip setuptools wheel poetry

# 6) Clean up apt caches
RUN apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/*

# 7) Copy repo contents
COPY ./ /spectralmc

# 8) Work in /spectralmc
WORKDIR /spectralmc

# Use the binary build pyproject (torch 2.7.1+cu128)
RUN cp pyproject.binary.toml pyproject.toml

# 9) Install dependencies system-wide via Poetry
RUN if [ "$BUILD_FROM_SOURCE" = "true" ]; then \
    poetry install --no-interaction --with dev; \
  else \
    poetry install --no-interaction --with dev; \
  fi

# 9.25) Create cache directories
RUN mkdir -p \
    /opt/spectralmc/cache \
    /opt/spectralmc/mypy_cache \
    /opt/spectralmc/pytest_cache \
    /opt/spectralmc/ruff_cache \
    /opt/pycache

# 9.5) Generate protobuf code into /opt/spectralmc_proto
RUN mkdir -p /opt/spectralmc_proto && \
    protoc \
        --python_out=/opt/spectralmc_proto \
        --proto_path=/spectralmc/src/spectralmc/proto \
        /spectralmc/src/spectralmc/proto/*.proto && \
    echo '"""Generated protobuf modules for SpectralMC."""' > /opt/spectralmc_proto/__init__.py && \
    python -c "import site; from pathlib import Path; pth = Path(site.getsitepackages()[0]) / 'spectralmc-proto.pth'; pth.write_text('/opt/spectralmc_proto\n/opt\n'); print(f'✓ Wrote {pth}')" && \
    ls -la /opt/spectralmc_proto/

# Verify protobuf generation
RUN PYTHONPATH=/opt:${PYTHONPATH:-} python -c "import spectralmc_proto.common_pb2; print('✓ Loaded protobuf modules')" || { echo "Failed to load protobuf"; exit 1; }

# 10) Block timeout usage with test commands but allow other timeout usage
RUN if [ -f /usr/bin/timeout ]; then mv /usr/bin/timeout /usr/bin/timeout.real; fi && \
    echo '#!/bin/sh\nif echo "$*" | grep -qE "(pytest|test-|test_)"; then\n  echo "❌ ERROR: Timeout commands with tests forbidden per CLAUDE.md"\n  echo "✅ USE: poetry run test-all [pytest-args]"\n  echo "📖 Tests complete when they complete - patience required"\n  exit 1\nfi\nexec /usr/bin/timeout.real "$@"' > /usr/bin/timeout && \
    chmod +x /usr/bin/timeout

# 11) Expose port 6006 (TensorBoard)
EXPOSE 6006

# 12) Default command to keep container alive for exec sessions
ENTRYPOINT ["/usr/bin/tini", "--"]

CMD ["sleep", "infinity"]
