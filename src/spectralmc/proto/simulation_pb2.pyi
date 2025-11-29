from google.protobuf import message

class SimulationParamsProto(message.Message):
    skip: int
    timesteps: int
    network_size: int
    batches_per_mc_run: int
    threads_per_block: int
    mc_seed: int
    buffer_size: int
    dtype: int

class BlackScholesConfigProto(message.Message):
    sim_params: SimulationParamsProto
    simulate_log_return: bool
    normalize_forwards: bool

class BoundSpecProto(message.Message):
    lower: float
    upper: float

class OptionContractProto(message.Message):
    X0: float
    K: float
    T: float
    r: float
    d: float
    v: float

class PricingResultsProto(message.Message):
    call_price: float
    put_price: float
    underlying: float
    forward: float
    gamma: float

class SobolConfigProto(message.Message):
    dimensions: int
    seed: int
    domain_bounds: dict[str, BoundSpecProto]
