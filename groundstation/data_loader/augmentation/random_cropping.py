import random
import torch
import numpy as np


class RandomCropping(object):
    def __init__(self, keep_percentage=.5, allow_empty_crop=False):
        self.keep_percentage = keep_percentage
        self.allow_empty_crop = allow_empty_crop

    def random_crop(self, img, width, height, bboxes):
        """

        find a random crop, so at least one object is in it
        """
        assert img.shape[0] >= height
        assert img.shape[1] >= width
        # assert len(bboxes) > 0

        _bboxes = np.array(bboxes)

        runs = 1000
        if len(bboxes) == 0:
            runs = 1

        for _ in range(runs):
            x = random.randint(0, img.shape[1] - width)
            y = random.randint(0, img.shape[0] - height)
            x_2 = x + width
            y_2 = y + height

            is_object_in_crop = np.logical_and(np.logical_and(_bboxes[:, 0] < x_2,
                                              _bboxes[:, 2] > x),
                               np.logical_and(_bboxes[:, 1] < y_2,
                                              _bboxes[:, 3] > y))

            if is_object_in_crop.any():
                break

        if not self.allow_empty_crop and not is_object_in_crop.any():
            raise Exception("No suitable crop")

        img = img[y:y + height, x:x + width]
        return x, y, img

    def __call__(self, sample):
        img, annots = sample['img'], sample['annot']

        h, w = img.shape[:2]
        x, y, img = self.random_crop(img, int(self.keep_percentage * w), int(self.keep_percentage * h), annots[:, :4])

        new_h, new_w = img.shape[:2]

        new_annotations = []

        for annot in annots:
            annot[0] = min(max(0, annot[0] - x), new_w-1)
            annot[1] = min(max(0, annot[1] - y), new_h-1)
            annot[2] = min(max(0, annot[2] - x), new_w-1)
            annot[3] = min(max(0, annot[3] - y), new_h-1)

            if (annot[2] - annot[0]) * (annot[3] - annot[1]) <= 0:
                # the annotation is outside of the crop
                continue

            new_annotations.append(annot)

        if len(new_annotations) > 0:
            new_annotations = torch.from_numpy(np.stack(new_annotations))
        else:
            new_annotations = torch.zeros((0, 5))

        sample['img'] = img
        sample['annot'] = new_annotations

        return sample