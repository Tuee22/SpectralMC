# File: documents/product/notebook_content_summary.md
# Notebook Content Migration

**Status**: Reference only  
**Supersedes**: Removed exploratory notebooks  
**Referenced by**: documents/product/index.md

> **Purpose**: Preserve durable guidance from removed notebooks without reintroducing notebook artifacts.
> **📖 Authoritative Reference**: [../documentation_standards.md](../documentation_standards.md)

All exploratory Jupyter notebooks have been removed. The essential guidance below captures the durable content previously stored in notebooks.

## GPU SDE Simulation (formerly `SDE_Simulation_with_CuPy.ipynb`)
- Euler–Maruyama discretization for SDEs / GBM with drift `mu` and volatility `sigma`.
- GPU path simulation uses CuPy RNG for large normal draws and Numba CUDA kernels for per-path evolution (`idx = cuda.grid(1)`, blocks/threads configured from total path count).
- Emphasized reasons for GPU-only workflow: avoids PCIe copies, keeps Monte Carlo outputs resident for downstream ML, and achieves orders-of-magnitude faster RNG + kernel execution versus CPU.
- Notes on CUDA execution model: grid/block sizing, JIT warmup cost, and memory footprint awareness when generating multi-GB normal matrices.

## GBM CVNN Training Demo (formerly `test_gbm_train.ipynb`)
- Workflow: define `BlackScholesConfig`, create Sobol sampler, run GPU GBM simulation, FFT the payoff spectrum, train `GbmCVNNPricer`, then infer prices.
- TensorBoard usage: logdir per run; launch via `tensorboard --logdir <path>` to visualize `Loss/train`.
- Training config guidance: small batch counts for quick smoke tests; increase `batches_per_mc_run` or network size if GPU memory allows.
- Inference flow: `predict_price` returns a `Result` whose `Success` case yields the real component of the 0-frequency bin; non-trivial imaginary parts indicate an under-trained model.

## Removed Scratch Notebooks
- `async_normals.ipynb`, `bs_sampling_test.ipynb`, `cvnn.ipynb`, `discrete_fourier_transform.ipynb`, `error_duplication.ipynb`, `error_investigation.ipynb`, `gbm.ipynb`, `sobol_sampler.ipynb`, `test_gbm_train.ipynb` contained exploratory code or partial experiments now superseded by the summaries above and existing white papers in `documents/`.
