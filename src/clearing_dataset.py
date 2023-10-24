import shutil

from sklearn.ensemble import IsolationForest
import numpy as np
import pandas as pd
import os
import sys
import logging
from ast import literal_eval

_log_format = "%(asctime)s\t%(levelname)s\t%(name)s\t" \
              "%(filename)s.%(funcName)s " \
              "line: %(lineno)d | \t%(message)s"
logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter(_log_format))
logger.setLevel(logging.INFO)
logger.addHandler(handler)
logger.propagate = False


class Params:
    name_model = "swin_base_patch4_window7_224"
    path_to_save_result = "./results/clustering_result"
    visualize = True
    path_to_dataset = "D:\\projects_andrey\\cv_made_dataset\\vk-made-sports-image-classification"
    path_to_features = "./results/features_extractions/result_swin_base_patch4_window7_224.csv"


params = Params()


def clearing_dataset():
    '''
    Функция для поиска выбросов в датасете с помощью IsolationForest
    На вход подается предварительно извлеченные фичи из изображений с помощью нейронной сети

    '''
    logger.info(f"Read dataframe: {params.path_to_features}")
    df = pd.read_csv(params.path_to_features, sep=";")

    df['features_images'] = df['features_images'].apply(literal_eval)
    features = list(df['features_images'])
    logger.info(f"Start training IsolationForest...")

    clf = IsolationForest(max_samples=100,
                          random_state=1,
                          contamination='auto')
    preds = clf.fit_predict(features)
    logger.info(f"Predicts: {np.unique(preds, return_counts=True)}")
    os.makedirs(params.path_to_save_result, exist_ok=True)
    df = df.drop(['features_images'], axis=1)
    df["blowout"] = preds
    if params.visualize:
        path_to_save_result = os.path.join(params.path_to_save_result, "visualize")
        if params.visualize and os.path.exists(path_to_save_result):
            shutil.rmtree(path_to_save_result)

        for ind, row in df.iterrows():
            if row["blowout"] == -1:
                path_to_save = os.path.join(path_to_save_result, row["label"])
                os.makedirs(path_to_save, exist_ok=True)
                shutil.copy(os.path.join(params.path_to_dataset, 'train', row["image_id"]),
                            os.path.join(path_to_save, row["image_id"]))

    df.to_csv(os.path.join(params.path_to_save_result, f'result_{params.name_model}.csv'), index=False)

    logger.info("Done!")


if __name__ == "__main__":
    clearing_dataset()
