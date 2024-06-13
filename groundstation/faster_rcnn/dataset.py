import torch
import numpy as np


def collater(data):
    imgs = [s['img'] for s in data]
    annots = [s['annot'] for s in data]

    if 'scale' in data[0].keys():
        scales = [s['scale'] for s in data]
    else:
        scales = None

    img_ids = [s['img_id'] for s in data]

    shapes = np.stack([list(img.shape) for img in imgs])
    all_same_shape = (shapes == shapes[0].reshape(1, -1)).all()

    # now we want to stack the images into one tensor for this all images have to be equally large

    if not all_same_shape:
        largest_shape = (np.max(shapes[:, 0]), np.max(shapes[:, 1]), np.max(shapes[:, 2]))

        new_img = np.zeros((len(shapes), largest_shape[0], largest_shape[1], largest_shape[2]))
        for idx, img in enumerate(imgs):
            new_img[idx, :img.shape[0], :img.shape[1], :img.shape[2]] = img
        imgs = torch.from_numpy(new_img)
    else:
        imgs = torch.from_numpy(np.stack(imgs, axis=0))

    imgs = imgs.permute(0, 3, 1, 2).float()
    annots = [annot.float() for annot in annots]

    ret = {'img': imgs, 'annot': annots, 'img_id': img_ids}

    if scales is not None:
        ret['scale'] = scales

    return ret






