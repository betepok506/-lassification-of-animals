import json
import torchvision.transforms as transforms


def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def validate_transform(img_size):
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(img_size),
        transforms.ToTensor(),
    ])


def get_dict_classes(path_to_classes):
    with open(path_to_classes, 'r', encoding='utf-8') as f:
        encode_classes2index = json.load(f)

    decode_index2classes = {}
    for key in encode_classes2index.keys():
        decode_index2classes[encode_classes2index[key]] = key

    return encode_classes2index, decode_index2classes
