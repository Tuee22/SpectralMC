# Spectral Monte-Carlo Learning

## Overview

Spectral Monte Carlo is a robust online ML method for training CVNNs via Monte-Carlo data. It is inspired by techniques in Reinforcement Learning, particularly policy gradient methods. SpectralMC is useful for accelerating traditional Monte-Carlo methods, whenever the desired value from the MC sample is an expected function of the simulated distribution.

SpectralMC is particularly relevant to the field of Quantitative Finance, where it is standard to model markets using stochastic processes. Traditional Monte-Carlo methods for finance, generally involving a discretized simulation of SDEs, is a powerful technique for modelling financial instruments, however it is very compute intensive. 

Practically, SpectralMC can dramatically reduce the compute costs associated with running mark-to-market and risk metrics on a book of exotic derivatives. Rather than a large distributed valuation service with tens of thousands of cores, SpectralMC uses a small GPU cluster to provide continuous real-time pricing and risk for a large portfolio of exotics. As a fully online ML method, SpectralMC performs training and inference continuously. All MC simulations used for training are performed directly on the GPU. The compute capacity of the GPU cluster can be sized in order to achieve arbitrary inference accuracy with respect to the underlying Monte-Carlo methods.

SpectralMC is written entirely in Python using CUDA-native libraries (cupy, numba, pytorch). Bindings are also provided for quantlib.

## Key Features
- **Monte Carlo Simulation**: Generates a finite sample from a parametric distribution, given an arbitrary parameter vector. This is done directly on the GPU.
- **Fourier Transform**: Employs Fast Fourier Transform (FFT) on the sample data to estimate its characteristic function.
- **CVNN Training**: Updates the parameters of a complex-valued neural network to approximate the estimated characteristic function, using the arbitrary parameter vector as network inputs.
- **CVNN Inference**: Using the arbitrary parameter vector as network inputs, the CVNN produces an estimate of the distribution which would result from MC simulation from those parameters. Expressed as a complex-valued characteristic function, this can be used for computing desired distributional quantities (mean, moments, quantiles, CTE, etc).

## Getting Started
### Prerequisites
- **Nvidia GPU**: All code is CUDA native, so you will need an Nvidia GPU & drivers.
- **Docker**: Docker and Docker Compose are used to containerize the application, ensuring consistency across different development and production environments.
- **Nvidia Container Toolkit**: Needed to make the GPU available inside the container.

### Installation
Clone this repository using:
```bash
git clone https://github.com/yourusername/spectral-monte-carlo-learning.git
cd spectral-monte-carlo-learning
```
## Contributing

We welcome contributions to the SpectralMC project. If you would like to contribute, please fork the repository and submit a pull request. For substantial changes, please open an issue first to discuss what you would like to change.

## License

Distributed under the MIT License. See LICENSE for more information.

## Contact

Your Name - matt@resolvefintech.com
Project Link: https://github.com/Tuee22/SpectralMC
