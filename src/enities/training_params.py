from dataclasses import dataclass, field


@dataclass()
class TrainingParams:
    epochs: int
    lr: float
    batch_size: int
    num_workers: int
    img_size: tuple
    path_save_checkpoint: str
    path_to_classes: str
    name_model: str
    pretrained: bool
    path_to_weights: None or str
    path_to_dataset: str


    # output_metric_path: str
    # output_transformer_path: str
    # output_model_path: str
    # model_type: str = field(default="RandomForestRegressor")
    # random_state: int = field(default=42)