import numpy as np
import torch

class_map = {
    0: -1,
    1: 1,
    2: 1,
    3: 2,
    4: 2,
    5: 2,
    6: 2,
    7: 2,
    8: 2,
    9: 2,
    10: 2,
    11: -1
}

class SimplifyVisDrone(object):
    def __init__(self, map_classes=False):
        print("! Use only pedestrian an vehicles as classes !")

    def __call__(self, sample):
        annots = sample['annot']

        annots = annots.numpy()

        classes = annots[:, 4]

        def map(x):
            return class_map[x]

        classes = np.fromiter((map(xi) for xi in classes), classes.dtype)
        annots[:, 4] = classes

        annots = annots[annots[:, 4] >= 0]

        sample['annot'] = torch.from_numpy(annots)

        return sample

