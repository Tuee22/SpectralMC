
# Deep Complex‑Valued Set Encoders for Dupire Local Volatility Option Pricing

---

## Table of Contents
1. [Introduction](#introduction)  
2. [Feature Preparation with Random Fourier Features](#feature-preparation-with-random-fourier-features)  
3. [Architectures](#architectures)  
4. [End‑to‑End PyTorch Implementation Walk‑through](#end-to-end-pytorch-implementation-walk-through)  
5. [Extensions & Practical Tips](#extensions--practical-tips)  
6. [Conclusion](#conclusion)  

---

<a name="introduction"></a>
## 1  Introduction

### 1.1  Point Clouds as Local‑Vol Surfaces  
A **point cloud** is an unordered set of points

$$\{x_i\}_{i=1}^N \subset \mathbb{R}^d$$

Most deep‑learning layers assume a *fixed ordering*, but point clouds are **permutation invariant**:

$$f\big(\{x_{\pi(i)}\}\big) = f\big(\{x_i\}\big) \quad \forall\,\text{permutations } \pi$$

In option pricing under a **Dupire local‑volatility model**, each market surface is an irregular set of tuples  

$$x_i = (K_i,\; T_i,\; \sigma^{\text{loc}}_i)$$

where $K$ is strike, $T$ maturity, and $\sigma^{\text{loc}}$ the local volatility.  The number and placement of quotes varies day‑to‑day, so we need a model that handles **variable‑length, unordered inputs**.

### 1.2  Transformers for Sets  
Transformers replace recurrence with **self‑attention**, allowing every token to attend to every other.  If we drop positional encodings (or use content‑based encodings), attention becomes **order agnostic** (Set Transformer, 2019).

### 1.3  Complex‑Valued Networks & *modReLU*  
Our network outputs the complex **characteristic function**

$$\varphi(u) = \mathbb{E}\big[e^{i u X_T}\big]$$

so we keep tensors **complex throughout** and use the *modReLU* activation:

```python
def modrelu(z, b):
    r = torch.abs(z)
    return torch.relu(r + b) * torch.exp(1j * torch.angle(z))
```

---

<a name="feature-preparation-with-random-fourier-features"></a>
## 2  Feature Preparation with Random Fourier Features

### 2.1  Motivation  
Raw coordinates make it hard for a network to learn high‑frequency structure.  **Random Fourier Features (RFF)** embed inputs in a richer basis:

$$\phi_{\boldsymbol\omega}(x) = e^{i\,\boldsymbol\omega^{\top} x}, \quad \boldsymbol\omega \sim \mathcal{N}(0,\Sigma)$$

### 2.2  Complex RFF for $(K,T)$  
Work in **log‑moneyness** $m = \log(K/S_0)$ and maturity $T$.  
Draw $D$ frequencies $\{\omega_j\}$ from a suitable spread.  
Define  

$$\text{RFF}(m,T) = \big[e^{i\omega_1 m},\; e^{i\omega_1 T},\;\dots, e^{i\omega_D T}\big]$$

Concatenate the scalar local vol:

$$x_i = \sigma^{\text{loc}}_i \;\otimes\; \text{RFF}(m_i,T_i) \;\in\; \mathbb{C}^{2D}$$

### 2.3  Minimal PyTorch Snippet
```python
class ComplexRFF(nn.Module):
    def __init__(self, d=12, base=1.0):
        super().__init__()
        freqs = base * torch.logspace(0, 2, d)
        self.register_buffer("freqs", freqs)

    def forward(self, m, T):
        phase_m = m.unsqueeze(-1) * self.freqs
        phase_t = T.unsqueeze(-1) * self.freqs
        return torch.cat([torch.exp(1j*phase_m),
                          torch.exp(1j*phase_t)], dim=-1)
```

---

<a name="architectures"></a>
## 3  Architectures

### 3.1  Complex PointNet / DeepSets
1. **Shared $\phi$** – complex MLP with *modReLU*.  
2. **Pooling** – mean/sum across points ⇒ permutation invariance.  
3. **Global $\rho$** – complex MLP mapping pooled vector to latent code $h$.

### 3.2  Complex Set Transformer  
Self‑attention weights  

$$\alpha_{ij} = \text{softmax}\big(\operatorname{Re}(Q_i K_j^{\dagger})\big)$$

give explicit pair interactions; Pooling‑by‑Multi‑Head Attention extracts global code.

### 3.3  Trade‑offs  

|               | PointNet | Set Transformer |
|---------------|----------|-----------------|
| Complexity    | $\mathcal{O}(N)$ | $\mathcal{O}(N^2)$ |
| Interactions  | implicit | explicit |
| Simplicity    | easier   | heavier |

---

<a name="end-to-end-pytorch-implementation-walk-through"></a>
## 4  End‑to‑End PyTorch Walk‑through

```python
# ComplexLinear, ModReLU, ComplexPointNet, CharFuncHead defined as in text …
```

```python
def fake_batch(B=16, N=40, D=12):
    m   = torch.randn(B, N)
    T   = 2.0 * torch.rand(B, N)
    sig = 0.1 + 0.4 * torch.rand(B, N)
    rff = ComplexRFF(D)
    pts = sig.unsqueeze(-1) * rff(m, T)          # (B,N,2D)
    target = torch.randn(B, 32, dtype=torch.cfloat)
    return pts, target

model = nn.Sequential(
    ComplexPointNet(point_dim=24),
    CharFuncHead()
)
opt = torch.optim.Adam(model.parameters(), 1e-3)
for step in range(200):
    x, t = fake_batch()
    y = model(x)
    loss = torch.mean(torch.abs(y - t) ** 2)
    opt.zero_grad(); loss.backward(); opt.step()
```

---

<a name="extensions--practical-tips"></a>
## 5  Extensions & Tips
* Martingale penalty: add $|\varphi(0)-1|^2$.  
* Frequency curriculum: start low‑freq, add highs progressively.  
* Complex LayerNorm: normalize magnitudes only.  
* Cross‑attention: query the vol set with target $(K^*,T^*)$.

---

<a name="conclusion"></a>
## 6  Conclusion
Fully complex, permutation‑invariant encoders built on **PointNet** or **Set Transformer**, combined with **Random Fourier Features** and *modReLU*, offer an elegant path to neural Dupire pricers that output characteristic functions.

