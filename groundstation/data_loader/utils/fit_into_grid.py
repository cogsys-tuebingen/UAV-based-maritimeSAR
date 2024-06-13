import numpy as np
import torch

class FitIntoGrid(object):
    def __init__(self, grid_size):
        self.grid_size = grid_size

    def __call__(self, sample):
        image, annots, img_ids = sample['img'], sample['annot'], sample['img_id']
        height, width, _ = image.shape

        new_width = width + (self.grid_size - (width % self.grid_size) if width % self.grid_size > 0 else 0)
        new_height = height + (self.grid_size - (height % self.grid_size) if height % self.grid_size > 0 else 0)

        new_image = np.zeros((new_height, new_width, 3))
        new_image[0:height, 0:width] = image.numpy()

        sample['img'] = torch.from_numpy(new_image)

        return sample