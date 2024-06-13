import numpy as np
import cv2
import torch


class CenterAffine(object):
    """
    Affine Transform for CenterNet
    """

    def __init__(self, border, output_size):
        """
        Args:
            border(int): border size of image
            output_size(tuple): a tuple represents (width, height) of image
        """
        self.border = border
        self.output_size = output_size

    def _get_warped_annots(self, bboxes, affine, w, h):
        coords = bboxes[:, :4].reshape(-1, 2)
        aug_coords = np.column_stack((coords, np.ones(coords.shape[0])))
        coords = np.dot(aug_coords, affine.T)
        coords[..., 0] = np.clip(coords[..., 0], 0, w - 1)
        coords[..., 1] = np.clip(coords[..., 1], 0, h - 1)

        bboxes[:, :4] = coords.reshape(-1, 4)

        # remove lost bboxes
        sizes = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
        bboxes = bboxes[sizes > 0]

        return bboxes

    def __call__(self, sample):
        """
        generate one `AffineTransform` for input image
        """
        img = sample['img'].numpy()
        img_shape = img.shape[:2]
        w, h = self.output_size

        bboxes = []

        iterations = 1000
        if len(sample['annot']):
            # if there is no annotation in the input, we can speed up the augmentation
            iterations = 1

        for i in range(iterations):
            center, scale = self.generate_center_and_scale(img_shape)
            src, dst = self.generate_src_and_dst(center, scale, self.output_size)
            affine = cv2.getAffineTransform(np.float32(src), np.float32(dst))

            # test the transformation with the annotations
            bboxes = sample['annot'].numpy()
            bboxes = self._get_warped_annots(bboxes.copy(), affine, w, h)

            if len(bboxes) > 0:
                break

        # if len(bboxes) == 0:
        #     raise Exception("Could not find a transformation which fits")

        img = cv2.warpAffine(img, affine, self.output_size, flags=cv2.INTER_LINEAR)

        sample['img'] = torch.from_numpy(img)
        sample['annot'] = torch.from_numpy(bboxes)

        return sample

    @staticmethod
    def _get_border(border, size):
        """
        decide the border size of image
        """
        # NOTE This func may be reimplemented in the future
        i = 1
        size //= 2
        while size <= border // i:
            i *= 2
        return border // i

    def generate_center_and_scale(self, img_shape):
        r"""
        generate center and scale for image randomly

        Args:
            shape(tuple): a tuple represents (height, width) of image
        """
        height, width = img_shape
        center = np.array([width / 2, height / 2], dtype=np.float32)
        scale = float(max(img_shape))
        scale = scale * np.random.choice(np.arange(0.6, 1.4, 0.1))
        h_border = self._get_border(self.border, height)
        w_border = self._get_border(self.border, width)
        center[0] = np.random.randint(low=w_border, high=width - w_border)
        center[1] = np.random.randint(low=h_border, high=height - h_border)

        return center, scale

    @staticmethod
    def generate_src_and_dst(center, size, output_size):
        r"""
        generate source and destination for affine transform
        """
        if not isinstance(size, np.ndarray) and not isinstance(size, list):
            size = np.array([size, size], dtype=np.float32)
        src = np.zeros((3, 2), dtype=np.float32)
        src_w = size[0]
        src_dir = [0, src_w * -0.5]
        src[0, :] = center
        src[1, :] = src[0, :] + src_dir
        src[2, :] = src[1, :] + (src_dir[1], -src_dir[0])

        dst = np.zeros((3, 2), dtype=np.float32)
        dst_w, dst_h = output_size
        dst_dir = [0, dst_w * -0.5]
        dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
        dst[1, :] = dst[0, :] + dst_dir
        dst[2, :] = dst[1, :] + (dst_dir[1], -dst_dir[0])

        return src, dst