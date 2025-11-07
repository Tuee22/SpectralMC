from google.protobuf import message

# Enum values
ACTIVATION_KIND_MOD_RELU: int
ACTIVATION_KIND_Z_RELU: int
ACTIVATION_KIND_C_RELU: int

class PreserveWidthProto(message.Message): ...

class ExplicitWidthProto(message.Message):
    value: int

class WidthSpecProto(message.Message):
    preserve: PreserveWidthProto
    explicit: ExplicitWidthProto
    def WhichOneof(self, name: str) -> str | None: ...

class ActivationCfgProto(message.Message):
    kind: int
    bias: float

class LinearCfgProto(message.Message):
    width: WidthSpecProto
    bias: bool
    activation: ActivationCfgProto
    def HasField(self, name: str) -> bool: ...

class LayerCfgProto(message.Message): ...

class NaiveBNCfgProto(message.Message):
    eps: float
    momentum: float

class CovBNCfgProto(message.Message):
    eps: float
    momentum: float
    num_groups: int

class SequentialCfgProto(message.Message):
    layers: list[LayerCfgProto]

class ResidualCfgProto(message.Message):
    block: LayerCfgProto
    learnable_scale: bool

class CVNNConfigProto(message.Message):
    dtype: int
    layers: list[LayerCfgProto]
    seed: int
