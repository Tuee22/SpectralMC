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

# 5) Install pip for Python 3.12
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12

# 6) Upgrade pip, setuptools, wheel
RUN python -m pip install --upgrade pip setuptools wheel

# 7) Clean up apt caches
RUN apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/*

# 8) Install Poetry
# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3.12 -

# Configure Poetry to not use virtual environments
RUN poetry config virtualenvs.create false
    
# 9) Copy repo contents
COPY ./ /spectralmc

# 10) Work in /spectralmc
WORKDIR /spectralmc

# 11) install dependencies system-wide
RUN poetry install --with dev --no-interaction

# validate code with mypy
RUN mypy /spectralmc/src/spectralmc /spectralmc/tests --strict --no-incremental && rm -rf .mypy_cache

# 12) Expose port 6006 & 8888
EXPOSE 6006 8888

# 13) Default command to run Jupyter
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", \
     "--NotebookApp.token=", "--NotebookApp.password=", "--notebook-dir=/spectralmc"]