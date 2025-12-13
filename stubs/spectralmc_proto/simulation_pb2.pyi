from __future__ import annotations

class SimulationParamsProto:
    skip: int
    timesteps: int
    network_size: int
    batches_per_mc_run: int
    threads_per_block: int
    mc_seed: int
    buffer_size: int
    dtype: int
    def __init__(self) -> None: ...
    def CopyFrom(self, other: SimulationParamsProto) -> None: ...

class PathSchemeProto:
    PATH_SCHEME_LOG_EULER: int
    PATH_SCHEME_SIMPLE_EULER: int

class ForwardNormalizationProto:
    FORWARD_NORMALIZATION_NORMALIZE: int
    FORWARD_NORMALIZATION_RAW: int

class BlackScholesConfigProto:
    sim_params: SimulationParamsProto
    path_scheme: int
    normalization: int
    def __init__(self) -> None: ...
    def CopyFrom(self, other: SimulationParamsProto | BlackScholesConfigProto) -> None: ...

class BoundSpecProto:
    lower: float
    upper: float
    def __init__(self) -> None: ...
    def CopyFrom(self, other: BoundSpecProto) -> None: ...
