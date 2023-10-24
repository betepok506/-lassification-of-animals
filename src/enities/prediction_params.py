from dataclasses import dataclass, field


@dataclass()
class PredictionParams:
    path_to_save_images: str
    path_to_save_result: str
    path_to_dataset: str
    path_to_classes: str
    name_model: str
    pretrained: bool
    visualize_result: bool
    path_to_weights: str
    img_size: tuple


    # epochs: int
    # lr: float
    # batch_size: int
    # num_workers: int
    # img_size: tuple
    # path_save_checkpoint: str
    # path_to_classes: str
    # name_model: str
    # pretrained: bool
    # path_to_weights: None or str
    # path_to_dataset: str