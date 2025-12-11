from __future__ import annotations

class TrainingConfigProto:
    num_batches: int
    batch_size: int
    learning_rate: float
    def __init__(self) -> None: ...
