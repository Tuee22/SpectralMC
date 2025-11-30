from google.protobuf import message

from . import simulation_pb2, tensors_pb2

class TrainingConfigProto(message.Message):
    num_batches: int
    batch_size: int
    learning_rate: float

class StepMetricsProto(message.Message):
    step: int
    loss: float
    gradient_norm: float
    learning_rate: float
    batch_time_ms: float

class GbmCVNNPricerConfigProto(message.Message):
    cfg: simulation_pb2.BlackScholesConfigProto
    domain_bounds: simulation_pb2.SobolConfigProto
    checkpoint: tensors_pb2.ModelCheckpointProto
    optimizer_state: tensors_pb2.AdamOptimizerStateProto
    torch_cpu_rng_state: bytes
    torch_cuda_rng_states: list[bytes]
