from __future__ import annotations
from typing import Dict, List

class TensorStateProto:
    shape: List[int]
    dtype: int
    device: int
    data: bytes
    requires_grad: bool
    def __init__(self) -> None: ...
    def CopyFrom(self, other: TensorStateProto) -> None: ...

class AdamParamStateProto:
    step: int
    exp_avg: TensorStateProto
    exp_avg_sq: TensorStateProto
    def __init__(self, *, step: int = ...) -> None: ...
    def CopyFrom(self, other: AdamParamStateProto) -> None: ...

class AdamParamGroupProto:
    lr: float
    beta1: float
    beta2: float
    eps: float
    weight_decay: float
    amsgrad: bool
    def __init__(self, *, lr: float = ..., beta1: float = ..., beta2: float = ..., eps: float = ..., weight_decay: float = ..., amsgrad: bool = ...) -> None: ...

class AdamOptimizerStateProto:
    state: Dict[int, AdamParamStateProto]
    param_groups: List[AdamParamGroupProto]
    def __init__(self) -> None: ...
    def CopyFrom(self, other: AdamOptimizerStateProto) -> None: ...

class RNGStateProto:
    cpu_rng_state: bytes
    cuda_rng_states: List[bytes]
    torch_cpu_rng_state: bytes
    torch_cuda_rng_states: List[bytes]
    def __init__(self) -> None: ...
    def CopyFrom(self, other: RNGStateProto) -> None: ...

class ModelCheckpointProto:
    model_state: Dict[str, TensorStateProto]
    model_state_dict: Dict[str, TensorStateProto]
    optimizer_state: AdamOptimizerStateProto
    cpu_rng_state: bytes
    cuda_rng_states: List[bytes]
    rng_state: RNGStateProto
    torch_cpu_rng_state: bytes
    torch_cuda_rng_states: List[bytes]
    global_step: int
    def __init__(self) -> None: ...
    def CopyFrom(self, other: ModelCheckpointProto) -> None: ...
    def SerializeToString(self) -> bytes: ...
    def ParseFromString(self, data: bytes) -> None: ...
