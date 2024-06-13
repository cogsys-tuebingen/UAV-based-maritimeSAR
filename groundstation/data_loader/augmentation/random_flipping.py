import numpy as np
import torch


class RandomFlipping(object):
    """Random flipping"""

    def __call__(self, sample, flip_x=0.5):
        if np.random.rand() < flip_x:
            image, annots, img_ids = sample['img'].numpy(), sample['annot'].numpy(), sample['img_id']
            image = image[:, ::-1, :]

            rows, cols, channels = image.shape

            x1 = annots[:, 0].copy()
            x2 = annots[:, 2].copy()

            x_tmp = x1.copy()

            annots[:, 0] = cols - x2
            annots[:, 2] = cols - x_tmp

            sample['img'] = torch.from_numpy(image.copy())
            sample['annot'] = torch.from_numpy(annots)

        return sample