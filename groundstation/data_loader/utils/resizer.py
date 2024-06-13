import torch
import numpy as np
import cv2


class AdaptiveResizerSquare(object):
    """Convert ndarrays in sample to Tensors."""
    def __init__(self, max_size):
        self.max_size = max_size
        super(AdaptiveResizerSquare, self)

    def __call__(self, sample):
        image, annots, img_ids, meta = sample['img'].numpy(), sample['annot'].numpy(), sample['img_id'], sample['metadata']
        height, width, _ = image.shape
        common_size = np.min([meta, self.max_size])
        # print('common_size', common_size)

        if height > width:
            scale = common_size / height
            resized_height = common_size
            resized_width = int(width * scale)
        else:
            scale = common_size / width
            resized_height = int(height * scale)
            resized_width = common_size

        image = cv2.resize(image.astype(np.float32), (resized_width, resized_height))

        new_image = np.zeros((common_size, common_size, 3))
        new_image[0:resized_height, 0:resized_width] = image

        del image

        annots[:, :4] *= scale

        sample['img'] = torch.from_numpy(new_image)
        sample['annot'] = torch.from_numpy(annots)
        sample['scale'] = scale

        return sample


class ResizerSquare(object):
    """Convert ndarrays in sample to Tensors."""

    def __init__(self, common_size=512):
        self.common_size = common_size

    def __call__(self, sample):
        image, annots, img_ids = sample['img'].numpy(), sample['annot'].numpy(), sample['img_id']
        height, width, _ = image.shape

        if height > width:
            scale = self.common_size / height
            resized_height = self.common_size
            resized_width = int(width * scale)
        else:
            scale = self.common_size / width
            resized_height = int(height * scale)
            resized_width = self.common_size

        image = cv2.resize(image, (resized_width, resized_height))

        new_image = np.zeros((self.common_size, self.common_size, 3), dtype=image.dtype)
        new_image[0:resized_height, 0:resized_width] = image

        del image

        annots = annots.astype(np.float)
        annots[:, :4] *= scale
        # annots = annots.astype(np.int)

        sample['img'] = torch.from_numpy(new_image)
        sample['annot'] = torch.from_numpy(annots)
        sample['scale'] = scale

        return sample


class ResizerRectangle(object):
    """Convert ndarrays in sample to Tensors.
        Extend the image to rectangle. The longer side should have the common size """

    def __init__(self, common_size=512):
        self.common_size = common_size

    def __call__(self, sample):
        image, annots, img_ids = sample['img'].numpy(), sample['annot'].numpy(), sample['img_id']
        height, width, _ = image.shape

        if height > width:
            scale = self.common_size / height
            resized_height = self.common_size
            resized_width = int(width * scale)
        else:
            scale = self.common_size / width
            resized_height = int(height * scale)
            resized_width = self.common_size

        image = cv2.resize(image, (resized_width, resized_height))

        if annots.shape[-1] > 1:
            if annots.dtype == np.int:
                annots = annots.astype(np.float)
                annots[:, :4] *= scale
                annots = annots.astype(np.int)
            else:
                annots[:, :4] *= scale

        return {'img': torch.from_numpy(image), 'annot': torch.from_numpy(annots), 'scale': scale,
                'img_id': img_ids}


