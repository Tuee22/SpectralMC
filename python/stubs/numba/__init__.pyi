from typing import Any, Callable, ParamSpec, TypeVar

P = ParamSpec("P")
R = TypeVar("R")

class _CudaKernel:
    """
    Minimal kernel object supporting __getitem__ so we can do sde_simulation_cuda[blocks, threads](...).
    """

    def __getitem__(self, grid_block: tuple[int, int]) -> Callable[..., None]: ...

class cuda:
    @staticmethod
    def jit(signature: object = ..., **kwargs: object) -> _CudaKernel: ...
    @staticmethod
    def grid(dim: int) -> int: ...
    @staticmethod
    def as_cuda_array(obj: Any, stream: Any = ...) -> Any: ...
    @staticmethod
    def stream() -> Any: ...

def jit(
    signature: object = ..., **kwargs: object
) -> Callable[[Callable[P, R]], Callable[P, R]]: ...
