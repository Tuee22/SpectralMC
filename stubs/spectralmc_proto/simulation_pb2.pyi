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

class BlackScholesConfigProto:
    sim_params: SimulationParamsProto
    simulate_log_return: bool
    normalize_forwards: bool
    def __init__(self) -> None: ...
    def CopyFrom(self, other: SimulationParamsProto | BlackScholesConfigProto) -> None: ...

class BoundSpecProto:
    lower: float
    upper: float
    def __init__(self) -> None: ...
    def CopyFrom(self, other: BoundSpecProto) -> None: ...
