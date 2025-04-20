# Encoding Variable-Length Volatility Surfaces in Complex-Valued Neural Networks

Variable-length inputs of **(strike, maturity, implied volatility)** tuples pose a challenge for neural networks because the number of points can vary and the input is inherently **unordered**. In the context of asset pricing under a **risk-neutral measure**, these tuples represent an implied volatility surface — essentially a function of strike and maturity that encodes the risk-neutral distribution of the underlying asset’s future prices. This document surveys best practices for encoding such volatility surfaces into **complex-valued neural networks (CVNNs)**, ensuring the architecture can handle variable-length data while preserving the surface’s structure and arbitrage-free properties. We organize the discussion into theoretical techniques, applied architectures (especially from financial machine learning), and practical implementation notes.

---

## Theoretical Techniques for Variable-Length Input Encoding

### Permutation-Invariant Set Encodings (Deep Sets)

One fundamental approach to variable-length inputs is to treat the collection of tuples as an **unordered set** and design the network to be *permutation-invariant*. The Deep Sets framework (Zaheer et al., 2017) established that any permutation-invariant function can be decomposed into two functions: one that encodes each element and one that aggregates the encodings. In practice, a Deep Set model applies a shared encoding network \(\phi\) to each input tuple \((K,T,\text{IV})\) and then uses a symmetric aggregation (like sum, mean, or max) followed by another network \(\rho\) to produce a fixed-size representation or output. Formally,
\[
f(\{x_1,\dots,x_N\}) = \rho\Big(\sum_{i=1}^N \phi(x_i)\Big),
\]
which is invariant to any permutation of the inputs. This strategy naturally handles variable \(N\) (number of option quotes) and does not assume a particular ordering of strikes or maturities.

Using sum or average pooling as the aggregation ensures **inference on an arbitrary number of input tuples is possible**. Deep Sets are universal function approximators for set functions under mild conditions. However, the choice of aggregation can affect performance. In volatility surface encoding, sum/average pooling is typically used (sometimes max pooling can capture extremes, but for capturing the “shape,” sum or mean is more natural). One advantage for implied volatility data is that we can easily incorporate domain knowledge into \(\phi\) — for instance, using log-moneyness \(\log(K/F)\) or \(\sqrt{T}\) transforms. The Deep Sets approach has been proposed for portfolio aggregation and yield curve modeling in finance and generalizes well to volatility surfaces.

### Sequence and Recurrent Encodings

An alternative to explicit permutation-invariance is to impose an ordering on the data and use sequence models like RNNs or LSTMs. For instance, one could sort the volatility quotes by maturity and strike, then feed them into a recurrent network. However, a plain RNN will be **order-dependent**, so one must define a consistent, canonical ordering (e.g., ascending maturity, then ascending strike). This can work if you want to treat the data as a time series or if you want the model to exploit a notion of “term structure” as sequential.

Another idea is *Janossy pooling*, which approximates permutation invariance by training on multiple random orderings. This is less common in practice compared to simpler Deep Sets or attention-based pooling. Generally, RNNs aren’t the top choice for 2D surfaces like implied vol, but remain an option if you prefer sequential processing and are comfortable imposing an ordering.

### Attention-Based Pooling and Set Transformers

**Attention mechanisms** offer a powerful way to aggregate variable-length inputs while modeling interactions among them. The **Set Transformer** (Lee et al., 2019) introduced the idea of using self-attention to create permutation-invariant representations for sets. Instead of summation or max, an attention-based pooling can *learn* how to weight each point’s contribution.

In a Set Transformer, the input set is passed through self-attention blocks (SAB) to capture pairwise interactions among points. A final Pooling by Multihead Attention (PMA) layer produces a fixed-size output from a variable-size input. Because the queries in PMA are fixed and do not depend on input order, the representation is permutation-invariant. This is especially relevant for volatility surfaces since attention can automatically learn relationships among vol quotes, capturing the shape constraints (e.g., how short-term skew relates to long-term skew).

Another variant is using a **Transformer encoder** on the set of tuples, where each tuple \((K,T,\mathrm{IV})\) is treated as a “token.” Then you can use a special “[CLS]” token or an attention-based pooling to get a final summary. This can handle variable-length input without a fixed grid.

### Functional and Distributional Representations of the Surface

Rather than treating implied vol as just a set of discrete points, one can see it as a **continuous function** under the risk-neutral measure. Some approaches leverage known **parametric models** like SVI, then the parameters become a fixed-length encoding of the surface. Others train a **Variational Autoencoder (VAE)** to embed entire surfaces into a low-dimensional latent space, enforcing no-arbitrage constraints. One can also use **Neural Operators** like DeepONet or a **Graph Neural Operator (GNO)** to map any arbitrary set of quotes into a smooth surface. These methods handle variable inputs by design and can embed financial constraints (like arbitrage-free conditions).

### Complex-Valued Neural Network Considerations

Even though the input tuples \((K,T,\mathrm{IV})\) are real, we might use **complex-valued transformations** to leverage phase/magnitude operations or to naturally incorporate Fourier-based pricing methods. Techniques like Deep Sets, attention, or graph networks can be adapted to complex arithmetic. You represent real/imag parts separately, using split or modulus-based activations. This approach could be advantageous if the downstream tasks involve complex functions (e.g., characteristic functions for pricing). Note that many standard ML frameworks now support complex tensors, though specialized layers might need custom code.

---

## Applied Architectures in Finance

- **Deep Sets for Volatility Surfaces:** The idea of treating each market quote as an element of a set and applying a permutation-invariant aggregator (like sum) plus an MLP is straightforward and flexible. One can incorporate financial knowledge via transformations of \((K,T)\) into log-moneyness, etc. Then the aggregator produces a fixed-length representation that can feed downstream tasks like pricing or risk factor inference.

- **Convolutional Neural Networks on Vol Surface Grids:** Historically, many used 2D CNNs on a fixed strike–maturity grid, treating implied vol as an “image.” This captures local smoothness but requires a fixed input size and interpolation for missing points. Works by Stone (2018) and Dimitroff et al. (2018) used CNNs for fast calibration to volatility models.

- **Graph Neural Networks and Neural Operators:** More recent methods like **Graph Neural Operator (GNO)** can handle arbitrary sets of quotes by constructing a graph where each node is a strike–maturity point. The GNO learns to produce a smooth, arbitrage-free surface from variable inputs. This is especially useful for real-world data, which might have fewer quotes for certain maturities.

- **Attention and Transformer Models:** A **Set Transformer** or a standard Transformer encoder can handle the set of \((K,T,\mathrm{IV})\) tuples, producing a global embedding by multi-head attention. This can model inter-point relationships effectively, potentially capturing constraints across strikes and maturities without requiring a fixed grid.

- **Hypernetworks (HyperIV):** An approach where a smaller network (the hypernetwork) takes a handful of option quotes and outputs the parameters of another network that represents the entire vol surface. This can yield real-time inference and built-in arbitrage-free constraints.

- **Latent Factor Models and VAEs:** Another strategy is training a VAE to embed vol surfaces in a latent space, optionally ensuring no-arbitrage. This provides a compact representation that can be used for scenario generation or calibration. Some works decode from the latent space into parametric model parameters for guaranteed arbitrage-free surfaces.

- **Complex-Valued Approaches:** While fewer in finance, they can appear in tasks where the pricing function or characteristic function is fundamentally complex. One might also prefer complex arithmetic if using a Fourier-based method internally. The main difference is ensuring complex layers, activations, and backprop are well-defined.

### Practical Implementation Notes

- **Data Preparation**: Normalize or transform \((K,T)\) (e.g. to log-moneyness and \(\sqrt{T}\)). Possibly include global features (like spot, rate) in each tuple. If using a grid-based approach, you may need interpolation.

- **Batching**: Variable-length sets can be batched with padding and masking or using frameworks like PyTorch Geometric. Permutation invariance can be tested by permuting inputs and checking consistency.

- **Arbitrage Constraints**: One can incorporate no-arbitrage conditions via architecture constraints, penalties in the loss, or specialized parameterizations (e.g., SVI, GNO). This ensures physically valid surfaces for pricing tasks.

- **Complex Implementation**: If a CVNN is used, represent real/imag parts carefully, use appropriate initialization and complex activation (like modReLU). Some libraries (e.g. complexPyTorch) facilitate this.

- **Performance**: For typical vol surfaces (20–100 points), attention or GNN-based methods are feasible. If the data is extremely large, hierarchical or efficient operators may be needed.

- **Downstream Usage**: Whether for model calibration (mapping surface to parameters) or direct pricing (mapping surface + payoff to price), ensure the encoding preserves key structure so that results match standard pricing computations.

- **Open-Source Resources**:
  - **Operator Deep Smoothing** repo for GNO on IV surfaces
  - **HyperIV** code for real-time smoothing
  - Various Set Transformer / Deep Sets tutorials and official repos
  - Complex-valued layers in external libraries (complexPyTorch, etc.)

---

## References and Sources

1. **Zaheer et al. (2017).** *Deep Sets.* NeurIPS.  
2. **Stone (2018).** *Calibration of Rough Volatility Models via CNN.*  
3. **Dimitroff, Röder & Fries (2018).** *Applying CNNs to Vol Surface Calibration.*  
4. **Lee et al. (2019).** *Set Transformer: A Framework for Attention-based Permutation-Invariant Neural Networks.* ICML.  
5. **PointNet** *Qi et al. (2017).* CVPR.  
6. **GNO for IV surfaces** *Wichmann et al. (2024), Operator Deep Smoothing.*  
7. **HyperIV** *Yang et al. (2025).* Real-time Arbitrage-Free IV Smoothing.  
8. **Ning et al. (2022).** *Arbitrage-Free IV Surface Generation with VAEs.* arXiv.  
9. **Hammad (2024).** *Survey of Complex-Valued Neural Networks.* arXiv.  
10. **Murphy et al. (2019).** *Janossy Pooling.* arXiv.  
11. **S. G. & T. Arjovsky, (2016).** *modReLU for Unitary RNNs.*  
12. Various code repos:  
   - [Operator Deep Smoothing GitHub](https://github.com/rwicl/operator-deep-smoothing-for-implied-volatility)  
   - [HyperIV GitHub](https://github.com/qmfin/hyperiv)  
   - [BrianNingUT/ArbFreeIV-VAE](https://github.com/BrianNingUT/ArbFreeIV-VAE)  
   - [complexPyTorch](https://github.com/wavefrontshaping/complexPyTorch)

---

### Summary

By leveraging set-based, graph-based, or attention-based architectures, **complex-valued neural networks** can process a variable number of \((K,T,\mathrm{IV})\) tuples without forcing a fixed grid. The key is **permutation-invariant** or **mask-based** designs, plus domain-specific constraints like no-arbitrage. Modern methods (Deep Sets, Set Transformer, GNOs, hypernetworks) show strong results, with some open-source implementations available. Introducing complex-valued layers can align well with Fourier-based pricing or characteristic functions, though it requires specialized activation and gradient handling. Overall, these approaches allow flexible, structure-preserving encodings of implied volatility surfaces for downstream asset pricing under a risk-neutral measure.