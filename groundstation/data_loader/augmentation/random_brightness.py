import numpy as np
import torch


class RandomBrightness(object):
    """
        Randomly transforms image brightness.

        Brightness intensity is uniformly sampled in (intensity_min, intensity_max).
        - intensity < 1 will reduce brightness
        - intensity = 1 will preserve the input image
        - intensity > 1 will increase brightness

        See: https://pillow.readthedocs.io/en/3.0.x/reference/ImageEnhance.html
        """

    def __init__(self, intensity_min, intensity_max):
        """
        Args:
            intensity_min (float): Minimum augmentation
            intensity_max (float): Maximum augmentation
        """

        self.intensity_min = intensity_min
        self.intensity_max = intensity_max

    def __call__(self, sample):
        img = sample['img'].numpy()
        w = np.random.uniform(self.intensity_min, self.intensity_max)
        src_weight = 1 - w
        dst_weight = w
        src_img = 0

        if img.dtype == np.uint8:
            img = img.astype(np.float32)
            img = src_weight * src_img + dst_weight * img
            img = np.clip(img, 0, 255).astype(np.uint8)
        else:
            img = src_weight * src_img + dst_weight * img

        sample['img'] = torch.from_numpy(img)

        return sample