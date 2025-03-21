# Complex-Valued Neural Networks for Monte Carlo Pricing: A Summary

This document outlines a method for accelerating Monte Carlo (MC) pricing using Complex-Valued Neural Networks (CVNNs). Instead of directly learning payoffs, the CVNN learns the Discrete Fourier Transform (DFT) of a sample of MC outputs, which encodes the distribution of outcomes via its characteristic function.

## 1. Core Idea

- **Monte Carlo (MC) Pricing**: Time-consuming but highly flexible approach that simulates a payoff’s distribution.
- **Complex-Valued Neural Network (CVNN)**: A neural network whose parameters and activations are complex numbers, naturally supporting amplitude-phase (real-imag) representations.
- **Learning the DFT**: The DFT (characteristic function) of an MC sample encodes the entire distribution. Training a CVNN to predict the DFT makes it possible to recover any statistic or moment of the distribution.

## 2. Incremental Improvement

- Neural networks are universal function approximators. Given enough data and model capacity, a CVNN can approximate the pricing function arbitrarily well.
- By continually adding fresh data (i.i.d. MC samples), training can reduce noise in the DFT. In principle, the model’s accuracy can keep improving.
- Practical limits include overfitting, finite model capacity, and computational costs for training.

## 3. Comparing Different CVNN Architectures

- **Evaluation Metrics**:
  - Compare **mean squared error** (MSE) in the complex domain (real + imaginary parts) or real domain (inverted distribution).
  - Use distributional distances (e.g., Wasserstein, KL divergence) or tail-risk measures to assess how well each architecture captures extremes.
  - Consider model capacity (parameter count) to ensure fair comparisons.
- **Fairness**: Equalize parameter sizes and training conditions to isolate architectural differences.
- **Visual Tools**: Plot the model’s predicted prices vs. ground truth MC, look for alignment around the diagonal.

## 4. Potential for Replacing MC

- **Offline Training**: Once trained, the CVNN can generate real-time pricing for new inputs, offering huge speed-ups compared to repeated MC simulations.
- **Applications**: Particularly attractive for risk management, high-frequency trading, or interactive pricing tools where speed is critical.
- **Challenges**:
  - Ensuring tail risk is learned.
  - Validating that the output DFT corresponds to a valid probability distribution.
  - Handling complex training (regularization, initialization, etc.).
  - Maintaining reliability under new, possibly out-of-distribution scenarios.

## 5. Spectral vs. Time Domain Learning

- **Advantages** of Spectral (Fourier) Learning:
  - Potentially better capture of distribution shape, including high-frequency components.
  - CVNNs handle complex signals naturally.
- **Drawbacks**:
  - DFT of finite samples can be noisy.
  - Must handle aliasing, frequency resolution, and ensure sufficient bandwidth.
- **Practical Tools**: Techniques exist (Fourier transforms, specialized layers) for stable training in the spectral domain.

## 6. Implementation Details

- **Complex Loss Functions**: Sum of squared errors for real and imaginary parts is common. Phase-aware losses can help ensure correct distribution shape.
- **Complex Backpropagation**: Real and imaginary weights are updated via Wirtinger calculus or equivalent. Standard optimizers (SGD, Adam) apply with minor modifications.
- **Regularization**: Possibly needed to ensure the network’s DFT remains a valid characteristic function (e.g., ϕ(0) = 1, positivity constraints in the inverse transform).

## 7. Conclusions

- A CVNN-based DFT approach can, in theory, match or exceed MC accuracy with sufficient training, offering near-instant inference in production.
- Success hinges on careful data generation, network design, and validation, especially for tail events.
- Properly implemented, such a model can replace or drastically reduce the need for real-time MC while maintaining high accuracy in pricing and risk measures.