"""
/spectralmc/python/spectralmc/sde.py

Generic SDE simulation using Numba + CuPy with partial type hints.
No placeholders or ignoring. Must pass mypy --strict.

All imports at top, from __future__ import annotations so no quotes needed.
Relies on specialized overload: normal(float, float, (int,int), Type[float32]) -> ndarray
"""

from __future__ import annotations

import math
from typing import Any
import numba
from spectralmc import cp


@numba.cuda.jit
def sde_simulation_cuda(
    result: cp.ndarray[Any, Any],
    random_numbers: cp.ndarray[Any, Any],
    mu: float,
    sigma: float,
    X0: float,
    T: float,
    N: int,
) -> None:
    """
    Numba kernel for dX = mu*X dt + sigma*X * dW.

    Args:
        result: shape (N, paths) storing final simulation
        random_numbers: shape (N-1, paths) storing increments
        mu: drift
        sigma: vol
        X0: initial value
        T: horizon
        N: timesteps
    """
    idx = numba.cuda.grid(1)
    if idx < result.shape[1]:
        dt = T / float(N)
        X = X0
        for i in range(1, N):
            dW = random_numbers[i - 1, idx]
            X += mu * X * dt + sigma * X * dW
            result[i, idx] = X


def simulate_sde(
    paths: int,
    N: int,
    mu: float,
    sigma: float,
    X0: float,
    T: float,
    threads_per_block: int = 256,
) -> cp.ndarray[Any, Any]:
    """
    Simulate an SDE with Euler-Maruyama on the GPU, returning cp.ndarray.

    Args:
        paths: number of paths
        N: timesteps
        mu: drift
        sigma: vol
        X0: initial
        T: horizon
        threads_per_block: threads per block

    Returns:
        Cupy ndarray of shape (N, paths), dtype=cp.float32
    """
    if paths <= 0 or N <= 1:
        raise ValueError("paths must be > 0 and N > 1.")

    dt_sqrt = math.sqrt(T / float(N))

    rng = cp.random.default_rng(42)
    # EXACT match for specialized overload:
    # normal(float, float, tuple[int,int], Type[numpy.float32]) -> ndarray
    random_numbers = rng.normal(0.0, dt_sqrt, (N - 1, paths), cp.float32)

    result = cp.zeros((N, paths), dtype=cp.float32)
    result[0, :] = X0

    blocks = (paths + threads_per_block - 1) // threads_per_block
    random_dev = numba.cuda.as_cuda_array(random_numbers)
    result_dev = numba.cuda.as_cuda_array(result)

    sde_simulation_cuda[blocks, threads_per_block](
        result_dev, random_dev, mu, sigma, X0, T, N
    )
    return result
