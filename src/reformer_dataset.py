import os
import shutil
import json
import pandas as pd
from tqdm import tqdm

PATH_TO_OLD_DATASET = 'D:\\projects_andrey\\ru_code_contest\\photos'
PATH_TO_NEW_DATASET = 'D:\\projects_andrey\\ru_code_contest\\new_format_dataset'
if __name__ == "__main__":
    os.makedirs(PATH_TO_NEW_DATASET, exist_ok=True)
    list_name_img = []
    classes_images = []
    unique_classes = set()
    folder = 'train'

    for folder_cls in os.listdir(os.path.join(PATH_TO_OLD_DATASET, folder)):
        os.makedirs(os.path.join(PATH_TO_NEW_DATASET, folder), exist_ok=True)

        for name_img in tqdm(os.listdir(os.path.join(PATH_TO_OLD_DATASET, folder, folder_cls))):
            list_name_img.append(name_img)
            unique_classes.add(folder_cls)
            classes_images.append(folder_cls)

            shutil.copy(os.path.join(PATH_TO_OLD_DATASET, folder, folder_cls, name_img),
                        os.path.join(PATH_TO_NEW_DATASET, folder, name_img))

    classes2id = {k: v for v, k in enumerate(unique_classes)}
    id_classes = []
    for cls in classes_images:
        id_classes.append(classes2id[cls])

    with open(os.path.join(PATH_TO_NEW_DATASET, 'encode_classes.json'), 'w') as f:
        json.dump(classes2id, f)

    pd.DataFrame({'image_id': list_name_img, 'label': classes_images}).to_csv(
        os.path.join(PATH_TO_NEW_DATASET, f'{folder}.csv'), index=False)

    folder = 'test'
    os.makedirs(os.path.join(PATH_TO_NEW_DATASET, folder), exist_ok=True)
    list_name_img.clear()
    for name_img in tqdm(os.listdir(os.path.join(PATH_TO_OLD_DATASET, folder))):
        list_name_img.append(name_img)

        shutil.copy(os.path.join(PATH_TO_OLD_DATASET, folder, name_img),
                    os.path.join(PATH_TO_NEW_DATASET, folder, name_img))
    pd.DataFrame({'image_id': list_name_img}).to_csv(
        os.path.join(PATH_TO_NEW_DATASET, f'{folder}.csv'), index=False)