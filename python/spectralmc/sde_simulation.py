# python/spectralmc/sde_simulation.py

"""
Demonstrates a GPU-accelerated SDE simulation (Euler-Maruyama) via CuPy and Numba.
Adapted from SDE_Simulation_with_CuPy.ipynb. Minimal leftover ignores removed.
"""

from __future__ import annotations

import cupy as cp  # type: ignore[import-untyped]
from numba import cuda  # type: ignore[import-untyped]


@cuda.jit
def sde_simulation_cuda(
    result: cp.ndarray,
    random_numbers: cp.ndarray,
    mu: float,
    sigma: float,
    X0: float,
    T: float,
    N: int,
) -> None:
    """
    Euler-Maruyama for 1D SDE, parallel over paths dimension.
    """
    idx = cuda.grid(1)
    if idx < result.shape[1]:
        X = X0
        dt = T / N if N > 0 else 0.0
        for i in range(1, N):
            dW = random_numbers[i - 1, idx]
            X += mu * X * dt + sigma * X * dW
            result[i, idx] = X


def run_sde_simulation(
    paths: int,
    N: int,
    mu: float,
    sigma: float,
    X0: float,
    T: float,
    threads_per_block: int = 256,
) -> cp.ndarray:
    """
    High-level function to run the SDE on GPU, returning a Cupy array (N, paths).
    """
    dt_sqrt = (T / N) ** 0.5 if N > 0 else 0.0
    rng_mat = (
        cp.random.normal(0.0, dt_sqrt, size=(N - 1, paths))
        if N > 1
        else cp.zeros((0, paths))
    )
    result_gpu = cp.zeros((N, paths), dtype=cp.float32)
    result_gpu[0, :] = X0

    rng_mat_dev = cuda.to_device(rng_mat)
    result_dev = cuda.to_device(result_gpu)

    blocks_per_grid = (paths + threads_per_block - 1) // threads_per_block
    sde_simulation_cuda[blocks_per_grid, threads_per_block](
        result_dev, rng_mat_dev, mu, sigma, X0, T, N
    )
    cuda.synchronize()

    return result_dev.copy_to_host()
