from typing import Any


# this trick passes through any torch types we haven't specified from torchtyping
def __getattr__(name: str) -> Any: ...
