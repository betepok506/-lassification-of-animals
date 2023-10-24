from dataclasses import dataclass, field


@dataclass()
class SplittingParams:
    path_to_dataset: str
    test_size: float
    name_file: str
