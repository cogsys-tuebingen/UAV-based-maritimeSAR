import numpy as np
import torch


class AddMargin:
    """
        Adds black margin to the right and the bottom
    """
    def __init__(self, size=20):
        self.size = size

    def __call__(self, sample):
        image = sample['img']

        height, width, _ = image.shape

        new_image = np.zeros((height + self.size, width + self.size, 3))
        new_image[:height, :width] = image.numpy()

        sample['img'] = torch.from_numpy(new_image)

        return sample