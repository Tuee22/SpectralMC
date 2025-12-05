# File: documents/domain/index.md
# Domain Knowledge

**Status**: Reference only  
**Supersedes**: None  
**Referenced by**: documents/README.md

> **Purpose**: Index for SpectralMC domain and research whitepapers.
> **📖 Authoritative Reference**: [../documentation_standards.md](../documentation_standards.md)

## Overview

Scientific background, quantitative finance theory, and research papers underlying SpectralMC.

## Core Concepts

SpectralMC uses Complex-Valued Neural Networks (CVNNs) to learn characteristic functions from Monte Carlo simulations, enabling fast derivative pricing in quantitative finance.

## Whitepapers

### [SpectralMC Whitepaper](whitepapers/spectralmc_whitepaper.md)

Summary of CVNN approach for Monte Carlo pricing:

- Core methodology
- Incremental improvement approach
- Comparing architectures
- Potential for replacing Monte Carlo

**Key Topics**: CVNN, DFT, Monte Carlo acceleration

### [Characteristic Functions for Stochastic Processes](whitepapers/characteristic_function_for_stochastic_processes.md)

Mathematical theory of characteristic functions in stochastic process modeling.

**Key Topics**: Fourier analysis, stochastic processes, distributions

### [Deep Complex-Valued Set Encoders](whitepapers/deep_complex_valued_set_encoders.md)

Architecture research for complex-valued neural networks.

**Key Topics**: CVNN architectures, set encoding, permutation invariance

### [Imaginary Numbers: Unified Intuition](whitepapers/imaginary_numbers_unified_intuition.md)

Intuitive understanding of complex numbers in neural networks.

**Key Topics**: Complex numbers, phase-amplitude representation

### [Methodology Review](whitepapers/methodology_review.md)

Literature review of related methods in quantitative finance.

**Key Topics**: Neural network pricing, Fourier methods, literature review

### [Variable-Length CVNN Inputs](whitepapers/variable_length_cvnn_inputs.md)

Research on handling variable-length input sequences.

**Key Topics**: Sequence modeling, variable inputs, architecture extensions

## Related Documentation

- [Engineering Standards](../engineering/index.md) - Implementation practices
- [Product Documentation](../product/index.md) - Deployment and operations
- [PyTorch Facade](../engineering/pytorch_facade.md) - CVNN implementation patterns and determinism
