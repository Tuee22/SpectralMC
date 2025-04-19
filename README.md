# Spectral Monte-Carlo Learning

## Overview

Spectral Monte-Carlo (SpectralMC) is an online machine-learning method that trains complex-valued neural networks (CVNNs)using Monte-Carlo data. It draws on techniques from Reinforcement Learning, particularly policy gradient methods. This method is especially useful when the desired quantity is an expected function of a simulated distribution.

While useful in multiple domains, SpectralMC is especially relevant to the field of Quantitative Finance, where stochastic processes are widely used. Traditional Monte-Carlo-based solutions can be compute-intensive, sometimes requiring large compute clusters. By contrast, SpectralMC trains and performs inference continuously on the GPU to reduce overall computational load. This allows efficient training and inference accuracy, with a much smaller computational footprint than traditional Monte-Carlo methods.

## Key Features

- Monte-Carlo Simulation: Generates a finite sample from a parametric distribution directly on the GPU.
- Fourier Transform: Uses Fast Fourier Transform (FFT) to estimate the sample's characteristic function.
- CVNN Training: Updates the parameters of a complex-valued neural network to approximate the characteristic function.
- CVNN Inference: Produces an estimated distribution (via the characteristic function), enabling computation of means, moments, quantiles, and other metrics.

## Getting Started

## Prerequisites

- A ubuntu 24.04 machine with an NVIDIA GPU


## Installation

1. Use a Ubuntu 24.04 machine with an nvidia GPU. If you're on AWS, for example, you might choose a g5.xlarge instance, running Ubuntu 24.04.

2. Clone this repository:
```bash
git clone https://github.com/Tuee22/SpectralMC.git
cd SpectralMC
```

3. Run the provided installation script, which installs Docker, the NVIDIA drivers, and the NVIDIA Container Toolkit. For example:
```bash
./scripts/build_and_run_unit_tests.sh
```

- This will install Docker, the Nvidia drivers, and nvidia-docker2. When finished, reboot to fully activate the driver (the script will offer to do this for you, press Ctrl-C to stop it). After rebooting, run the script again, and it will build and start the container, and execute test cases.

4. To interact with the container manually:

```bash
cd docker
docker compose up -d
docker compose exec -it spectralmc bash
```

This will open a bash terminal, which will allow you to explicitly call the python library, eg

```bash
python -m spectralmc.gbm
```

There is also a jupyter notebook listening on port 8888, which you can access in your browser via localhost:8888.

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


