# File: documents/domain/whitepapers/characteristic_function_for_stochastic_processes.md
# Characteristic Functions for Stochastic Processes in a 2D Complex Representation

**Status**: Reference only  
**Supersedes**: Prior drafts on characteristic functionals  
**Referenced by**: documents/domain/index.md

> **Purpose**: Explain characteristic functionals for stochastic processes and their fixed-size representations for CVNNs.
> **📖 Authoritative Reference**: [../../documentation_standards.md](../../documentation_standards.md)

This document explores how **characteristic functions (CFs)**—Fourier transforms of probability distributions—can be extended to time-varying processes. We discuss the concept of **characteristic functionals**, which uniquely determine continuous-time stochastic processes, and describe practical 2D complex-valued representations that summarize marginal distributions over time. We then highlight how such representations can be used as fixed-size inputs for complex-valued neural networks (CVNNs), including ideas around wavelet or Fourier transforms along the time axis and learnable frequency selection.

---

## Gentle Introduction: Why Characteristic Functions Uniquely Define Distributions

A **characteristic function** (CF) of a real-valued random variable $X$ is defined as 

$$
\varphi_X(u)\;=\;\mathbb{E}\bigl[e^{\,i\,u\,X}\bigr].
$$

This function $(\varphi_X:\mathbb{R}\to\mathbb{C})$ captures the entire distribution of $X$ because its Taylor (or Fourier) expansion contains all the moments (if they exist), and more generally, it is **injective** with respect to the law of $X$. Concretely, no two distinct distributions can share the same characteristic function. This injectivity ensures that knowing $\varphi_X$ for all real $u$ completely determines the distribution of $X$. In fact, by inverting the characteristic function (via the inverse Fourier transform), one recovers the probability density (if it exists) or at least the cumulative distribution function of $X$.

When we extend from a single random variable to an entire continuous-time stochastic process $\{X(t)\}_{t\in[0,T]}$, we need a functional that captures all finite-dimensional distributions (i.e., the joint distribution at times $t_1,\dots,t_n$ for any $n$). The relevant notion is the **characteristic functional** $F[\phi]$:

$$
F[\phi] \;=\; \mathbb{E}\Bigl[\exp\!\Bigl(\,i \int_0^T \phi(t)\,X(t)\,\mathrm{d}t\Bigr)\Bigr].
$$

Under suitable assumptions, distinct path distributions **cannot** share the same $F[\phi]$. Thus, by analogy with the single-time case, the entire law of a continuous-time process can be recovered from its characteristic functional. Although we don’t typically evaluate $F[\phi]$ for all possible $\phi$, it remains a conceptual foundation for how these Fourier-based transforms generalize to infinite-dimensional spaces.

---

## Time-Varying Characteristic Function Grids for Fixed-Size Encoding

### CF Grid by Time and Frequency
A practical way to represent the evolving distribution $\{X(t)\}$ is to **sample** the characteristic function at a finite set of time points $\{t_1,\dots,t_M\}$ and frequencies $\{u_1,\dots,u_N\}$. We form a 2D array

$$
\Phi_{i,j} \;=\; \mathbb{E}\bigl[e^{\,i\,u_j\,X(t_i)}\bigr],
$$

which becomes an $M\times N$ matrix of complex numbers. Each row (indexed by $t_i$) is the CF of the marginal distribution of $X(t_i)$, evaluated at frequencies $u_1,\dots,u_N$.

- **Why it works**: For each time $t_i$, $\Phi_{i,\cdot}$ can be inverted (in principle) to recover that marginal distribution.  
- **Fixed size**: Regardless of the original number of simulation steps, the representation is an $M\times N$ complex matrix. You choose $M$ and $N$ based on desired resolution in time and frequency.  
- **Complex-valued**: Each $\Phi_{i,j}$ is a complex number, naturally usable by CVNNs which can exploit magnitude and phase.

### Extending to Joint Distributions
If one needed to encode **multi-time joint** distributions, one could sample a joint CF $\varphi_{t_1,\dots,t_m}(u_1,\dots,u_m)$ for some small $m$. But typically, storing all marginals $t_i$ plus partial correlation measures suffices for many tasks. The simpler 2D approach is widely used in CF-based modeling.

---

## Characteristic Functionals and Path Distributions

A rigorous approach uses the **characteristic functional** $F[\phi]$ mentioned above. Because we can’t store an infinite-dimensional object, we rely on discrete approximations (like the CF grid) or expansions (like the signature method). 

### Intuitive Reason for Uniqueness
- If two stochastic processes had the same functional $F[\phi]$ for all $\phi$, their finite-dimensional distributions must coincide, thus the processes have the same law.  
- This is the same principle that ensures $\varphi_X(u)$ is injective for single-time distributions. We’re just extending the exponent from $uX$ to $\int\phi(t)X(t)\,\mathrm{d}t$.

---

## Additional Representations: Time-Frequency or Wavelet

### Double Fourier / Time-Frequency
We can treat each row of the CF grid (for fixed $u_j$) as a time series $\{\varphi_{X(t_i)}(u_j)\}_{i=1..M}$. A Fourier transform or wavelet transform along $t$ can yield a **2D time-frequency** or time-scale representation. This can be beneficial if the distribution changes smoothly or exhibits multi-scale temporal dynamics.

### Learnable Frequency Grids
One could even learn the frequencies $\{u_j\}$ or wavelet scales via gradient-based optimization. This approach appears in *characteristic function distance* (CFD) setups for generative modeling, where the system learns which $u_j$ best discriminate distributions.

---

## CVNN Use Cases

### Feeding the CF Grid into a CVNN
A 2D array $\Phi_{i,j}\in\mathbb{C}$ can be fed to a complex-valued neural network. Real/imag parts act like separate channels, but the network’s weights and activations remain in $\mathbb{C}$. Such networks can exploit properties like complex multiplication/rotation, which aligns well with Fourier-based data.

### CF-GAN or PCF-GAN
Recent works (e.g. [Ansari et al., 2020](#ref-ansari), [Lou et al., 2023](#ref-lou)) train generative models to match **characteristic function** representations, ensuring they match the distribution at multiple times or frequencies. Such an approach can preserve marginals and even correlations if joint CFs are partially included.

---

## Why Characteristic Functions?

1. **Uniqueness**: A distribution is fully determined by its CF.  
2. **Fourier**: The CF is a natural Fourier transform of the probability measure. Convolution in real space becomes multiplication in CF space, aiding certain modeling tasks.  
3. **Complex Arithmetic**: CVNNs align neatly with complex inputs.  
4. **Discrete Approximation**: Even a modest grid of times/frequencies often suffices for distribution matching or feature extraction in ML pipelines.

---

## References and Sources

1. <a id="ref-fageot"></a>**Fageot, J. et al. (2014)**. *On the Continuity of Characteristic Functionals and Sparse Stochastic Modeling*. J. Fourier Analysis & Applications, **20**(6), 1179–1211.  
2. <a id="ref-lou"></a>**Lou, H. et al. (2023)**. *PCF-GAN: Generating Sequential Data via the Characteristic Function of Measures on the Path Space*. In NeurIPS 2023.  
3. **Chevyrev, I. & Lyons, T. (2016)**. *Characteristic Functions of Measures on Geometric Rough Paths*. Ann. Prob. **44**(6), 4049–4082.  
4. <a id="ref-ansari"></a>**Ansari, A.F. et al. (2020)**. *A Characteristic Function Approach to Deep Implicit Generative Modeling*. CVPR.  
5. **Askari Moghadam, R. & Sohrabi, A. (2017)**. *Complex-valued Neural Network and Normalized Fourier Transformation for Prediction of Time Series*. Asian J. Math. & Comp. Research, **21**(4), 179–192.  
6. **Addison, P.S. (2019)**. *The Illustrated Wavelet Transform Handbook: Introductory Theory and Applications in Science, Engineering, Medicine and Finance*. CRC Press.  
7. **Carrasco, M. & Nguyen, T. (2002)**. *Spectral GMM Estimation for Dynamic Models with Errors-in-Variables*.  
8. **Hammad, M. (2024)**. *Survey of Complex-Valued Neural Networks*.  

---

### Summary

By sampling **characteristic functions** (Fourier transforms) of a time-dependent random process at a finite set of times and frequencies, one obtains a **2D complex-valued** grid. Each row (a time slice) preserves the shape of that marginal distribution; each column (a frequency) captures how that mode evolves over time. Because characteristic functions uniquely determine distributions, this grid is an injective representation of the marginal laws over time. Additional transforms—such as wavelets—can further highlight multi-scale time behavior.  

Such **CF-based 2D arrays** are well-suited for **complex-valued neural networks (CVNNs)**, which can use magnitude and phase for learning tasks (e.g., generative modeling). Empirical results (e.g. CF-GAN, PCF-GAN) confirm that matching CF-based features can effectively learn or reproduce entire time-series distributions in an arbitrage-free or distribution-consistent manner.
