# typings/torch/backends/cudnn.pyi
"""
Tiny stub for :pymod:`torch.backends.cudnn`.

Only the attributes used by SpectralMC are declared.
"""

def version() -> int | None: ...

deterministic: bool
benchmark: bool
