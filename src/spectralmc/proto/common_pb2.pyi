from google.protobuf import message

# Enum values
PRECISION_FLOAT32: int
PRECISION_FLOAT64: int
DEVICE_CPU: int
DEVICE_CUDA: int
DTYPE_FLOAT32: int
DTYPE_FLOAT64: int
DTYPE_FLOAT16: int
DTYPE_BFLOAT16: int
DTYPE_COMPLEX64: int
DTYPE_COMPLEX128: int

class ModelVersionProto(message.Message):
    counter: int
    semantic_version: str
    parent_hash: str
    content_hash: str
    commit_timestamp: str
    commit_message: str

class TorchEnvProto(message.Message):
    python_version: str
    torch_version: str
    cuda_version: str
    cudnn_enabled: bool
    platform: str

class ArchitectureFingerprintProto(message.Message):
    cvnn_structure_hash: str
    total_parameters: int
    activation_kinds: str
