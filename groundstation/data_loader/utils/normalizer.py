import torch
import numpy as np


def normalize_img(img):
    img = img.astype(np.float32) / 255.

    mean = np.array([[[0.485, 0.456, 0.406]]])
    std = np.array([[[0.229, 0.224, 0.225]]])

    img = (img.astype(np.float32) - mean) / std

    return img


class Normalizer(object):
    def __init__(self):
        pass

    def __call__(self, sample):
        sample['img'] = torch.from_numpy(normalize_img(sample['img'].numpy()))

        return sample