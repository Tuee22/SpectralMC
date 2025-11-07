from google.protobuf import message
from . import common_pb2

class TensorStateProto(message.Message):
    shape: list[int]
    dtype: int
    device: int
    data: bytes
    requires_grad: bool

class AdamParamStateProto(message.Message):
    step: int
    exp_avg: TensorStateProto
    exp_avg_sq: TensorStateProto

class AdamParamGroupProto(message.Message):
    lr: float
    beta1: float
    beta2: float
    eps: float
    weight_decay: float
    amsgrad: bool

class AdamOptimizerStateProto(message.Message):
    state: dict[int, AdamParamStateProto]
    param_groups: list[AdamParamGroupProto]

class RNGStateProto(message.Message):
    torch_cpu_rng_state: bytes
    torch_cuda_rng_states: list[bytes]

class ModelCheckpointProto(message.Message):
    model_state_dict: dict[str, TensorStateProto]
    optimizer_state: AdamOptimizerStateProto
    rng_state: RNGStateProto
    global_step: int
    torch_env: common_pb2.TorchEnvProto
    architecture: common_pb2.ArchitectureFingerprintProto
