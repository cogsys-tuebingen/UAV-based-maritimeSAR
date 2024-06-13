import numpy as np
import torch
import cv2


class RandomScale(object):
    def __init__(self, scales):
        self.scales = scales

    def __call__(self, sample):
        img, annots = sample['img'].numpy(), sample['annot'].numpy()

        h, w = img.shape[:2]

        scale = np.random.choice(self.scales, 1)[0]
        img = cv2.resize(img, (int(w * scale), int(h * scale)))

        annots[:, 0:4] *= scale

        sample['img'] = torch.from_numpy(img)
        sample['annot'] = torch.from_numpy(annots)

        return sample