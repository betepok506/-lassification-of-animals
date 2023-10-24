import pandas as pd
import os
from sklearn.utils import shuffle
import numpy as np
import json
from src.enities.splitting_params import SplittingParams
import hydra
import sys
import logging

_log_format = "%(asctime)s\t%(levelname)s\t%(name)s\t" \
              "%(filename)s.%(funcName)s " \
              "line: %(lineno)d | \t%(message)s"
logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter(_log_format))
logger.setLevel(logging.INFO)
logger.addHandler(handler)
logger.propagate = False


@hydra.main(version_base=None, config_path='../configs', config_name='splitting_config')
def split_dataset(params: SplittingParams):
    logging.info(f"Path to dataset: {params.path_to_dataset}")
    df = pd.read_csv(os.path.join(params.path_to_dataset, params.name_file))
    classes = np.unique(df["label"])
    train_df = pd.DataFrame(
        data={
            "image_id": [],
            'label': []
        }
    )
    valid_df = pd.DataFrame(
        data={
            "image_id": [],
            'label': []
        }
    )
    logger.info(f"Start of splitting")
    encode_classes = {}
    for ind_cls, cls in enumerate(classes):
        encode_classes[cls] = ind_cls

        new_df = df[df['label'] == cls]
        new_df = new_df.drop(['label'], axis=1)
        new_df['label'] = np.ones(len(new_df), dtype=np.int8) * ind_cls

        X_train_ = new_df.sample(int((1 - params.test_size) * len(new_df)))
        X_test_ = new_df[~new_df.index.isin(X_train_.index)]
        train_df = pd.concat([X_train_, train_df])
        valid_df = pd.concat([X_test_, valid_df])

    train_df = shuffle(train_df)
    valid_df = shuffle(valid_df)
    logger.info(f"Information about classes is saved in: {os.path.join(params.path_to_dataset, 'encode_classes.json')}")
    with open(os.path.join(params.path_to_dataset, "encode_classes.json"), "w") as outfile:
        json.dump(encode_classes, outfile)

    train_df = train_df.astype({'label': 'int32'})
    valid_df = valid_df.astype({'label': 'int32'})

    train_df.to_csv(os.path.join(params.path_to_dataset, "train_.csv"), index=False)
    valid_df.to_csv(os.path.join(params.path_to_dataset, "valid_.csv"), index=False)
    logger.info(f"Path to saved: {params.path_to_dataset}")
    logger.info(f"Done!")


if __name__ == "__main__":
    split_dataset()

    # df = pd.read_csv(os.path.join(PATH_TO_DATASET, NAME_DATASET))
    # classes = np.unique(df["label"])
    # train_df = pd.DataFrame(
    #     data={
    #         "image_id": [],
    #         'label': []
    #     }
    # )
    # valid_df = pd.DataFrame(
    #     data={
    #         "image_id": [],
    #         'label': []
    #     }
    # )
    #
    # encode_classes = {}
    # for ind_cls, cls in enumerate(classes):
    #     encode_classes[cls] = ind_cls
    #
    #     new_df = df[df['label'] == cls]
    #     new_df = new_df.drop(['label'], axis=1)
    #     new_df['label'] = np.ones(len(new_df), dtype=np.int8) * ind_cls
    #
    #     X_train_ = new_df.sample(int((1 - test_size)*len(new_df)))
    #     X_test_ = new_df[~new_df.index.isin(X_train_.index)]
    #     train_df = pd.concat([X_train_, train_df])
    #     valid_df = pd.concat([X_test_, valid_df])
    #
    # train_df = shuffle(train_df)
    # valid_df = shuffle(valid_df)
    # with open(os.path.join(PATH_TO_DATASET, "encode_classes.json"), "w") as outfile:
    #     json.dump(encode_classes, outfile)
    #
    # train_df = train_df.astype({'label': 'int32'})
    # valid_df = valid_df.astype({'label': 'int32'})
    #
    # train_df.to_csv(os.path.join(PATH_TO_DATASET, "train_.csv"), index=False)
    # valid_df.to_csv(os.path.join(PATH_TO_DATASET, "valid_.csv"), index=False)
