import os
import pandas as pd
from tqdm import tqdm
import shutil
import torch
import cv2
import hydra
import logging
import sys

from src.models.classifications_models import get_model
from src.utils.utils import (
    get_dict_classes
)
from src.enities.prediction_params import PredictionParams
from src.utils.utils import validate_transform

_log_format = "%(asctime)s\t%(levelname)s\t%(name)s\t" \
              "%(filename)s.%(funcName)s " \
              "line: %(lineno)d | \t%(message)s"
logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter(_log_format))
logger.setLevel(logging.INFO)
logger.addHandler(handler)
logger.propagate = False


@hydra.main(version_base=None, config_path='../configs', config_name='predict_config')
def predict_pipeline(params: PredictionParams):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    logger.info(f"The currently used device: {device}")
    logger.info(f"Distribute images into catalogs: {params.visualize_result}")
    logger.info(f"Image size: {params.img_size}")

    encode_classes2index, decode_index2classes = get_dict_classes(params.path_to_classes)

    if params.visualize_result and os.path.exists(params.path_to_save_images):
        shutil.rmtree(params.path_to_save_images)

    if os.path.exists(params.path_to_weights):
        logger.info(f"Loading the model: {params.path_to_weights}")
        state_dict = torch.load(params.path_to_weights)

        if "model_state" not in state_dict:
            model_weights = state_dict
        else:
            model_weights = state_dict["model_state"]

        logger.info("The model has been loaded successfully")
    else:
        raise "No model weights found!"

    model = get_model(params.name_model,
                      len(encode_classes2index),
                      requires_grad=False,
                      model_weights=model_weights,
                      pretrained=params.pretrained).to(device)
    model.eval()
    test = pd.read_csv(os.path.join(params.path_to_dataset, 'test.csv'))
    test = list(test['image_id'])

    images_id = []
    predict_labels = []
    transform = validate_transform(params.img_size)
    for file_name in tqdm(test, desc="Prediction", ncols=120):
        path_to_file = os.path.join(params.path_to_dataset, 'test', file_name)
        image = cv2.imread(path_to_file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if transform is not None:
            image = transform(image)

        image = image.unsqueeze(0)
        image = image.to(device)
        output = model(image)
        predict_label = output.detach().cpu().argmax(dim=1).item()
        name_class = decode_index2classes[predict_label]

        if params.visualize_result:
            path_save_predict = os.path.join(params.path_to_save_images, name_class)
            os.makedirs(path_save_predict, exist_ok=True)
            shutil.copy(path_to_file, path_save_predict)

        images_id.append(file_name)
        predict_labels.append(name_class)

    pd.DataFrame({"image_id": images_id,
                  'label': predict_labels}).to_csv(os.path.join(params.path_to_save_result,
                                                                f'result_{params.name_model}.csv'), index=False)
    logger.info("Done!")


if __name__ == "__main__":
    predict_pipeline()
