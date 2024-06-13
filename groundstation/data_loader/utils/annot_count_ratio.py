import numpy as np
import torch


class AnnotCountRatio:
    """
        Limits the max number of annotations per sample
    """
    def __init__(self, ratio=1/10):
        self.ratio = ratio

    def __call__(self, sample):
        if self.ratio is not None:
            image, annot = sample['img'], sample['annot']
            max_count = max(1, int(self.ratio * len(annot)))

            if len(annot) > max_count:
                remove_count = len(annot) - max_count
                remove_idx = np.random.choice(list(range(len(annot))), size=remove_count, replace=False)

                new_image = image.clone()
                for bbox in annot[remove_idx]:
                    new_image[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])] = 0

                annot = np.delete(annot.numpy(), remove_idx, axis=0)

                sample['img'] = new_image
                sample['annot'] = torch.from_numpy(annot)

        return sample