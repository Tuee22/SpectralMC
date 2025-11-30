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
ENV PATH="/root/.local/bin:/usr/local/bin:$PATH"

# these are needed to ensure cupy works with newer gen GPUs
# (will jit for the local architecture if we need it)
ENV CUPY_COMPILE_WITH_PTX=1
ENV CUPY_NVCC_GENERATE_CODE=current

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
    && rm -rf /var/lib/apt/lists/*

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

# 9) Install dependencies system-wide via Poetry
RUN poetry install --no-interaction

# 10) Enforce CLAUDE.md testing policies: Block direct pytest and timeout commands
# Block direct pytest command (python -m pytest still works for Poetry scripts)
# Poetry installs to /usr/local/bin, so we override that location
RUN if [ -f /usr/local/bin/pytest ]; then mv /usr/local/bin/pytest /usr/local/bin/pytest.real; fi && \
    echo '#!/bin/sh\necho "❌ ERROR: Direct pytest forbidden per CLAUDE.md"\necho "✅ USE: poetry run test-all [pytest-args]"\necho "📖 Examples:"\necho "  poetry run test-all              # Run all tests"\necho "  poetry run test-all -v           # Verbose output"\necho "  poetry run test-all -m gpu       # GPU tests only"\necho "📖 Reference: CLAUDE.md testing policies"\nexit 1' > /usr/local/bin/pytest && \
    chmod +x /usr/local/bin/pytest

# Also block /usr/bin/pytest in case it exists
RUN if [ -f /usr/bin/pytest ]; then mv /usr/bin/pytest /usr/bin/pytest.real; fi && \
    echo '#!/bin/sh\necho "❌ ERROR: Direct pytest forbidden per CLAUDE.md"\necho "✅ USE: poetry run test-all [pytest-args]"\necho "📖 Reference: CLAUDE.md testing policies"\nexit 1' > /usr/bin/pytest && \
    chmod +x /usr/bin/pytest

# Block timeout usage with test commands but allow other timeout usage
RUN if [ -f /usr/bin/timeout ]; then mv /usr/bin/timeout /usr/bin/timeout.real; fi && \
    echo '#!/bin/sh\nif echo "$*" | grep -qE "(pytest|test-|test_)"; then\n  echo "❌ ERROR: Timeout commands with tests forbidden per CLAUDE.md"\n  echo "✅ USE: poetry run test-all [pytest-args]"\n  echo "📖 Tests complete when they complete - patience required"\n  exit 1\nfi\nexec /usr/bin/timeout.real "$@"' > /usr/bin/timeout && \
    chmod +x /usr/bin/timeout

# 11) Expose port 6006 & 8888
EXPOSE 6006 8888

# 12) Default command to run Jupyter
ENTRYPOINT ["/usr/bin/tini", "--"]

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", \
     "--NotebookApp.token=", "--NotebookApp.password=", "--notebook-dir=/spectralmc"]