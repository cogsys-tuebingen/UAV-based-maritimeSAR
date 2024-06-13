import numpy as np
import torch

class RandomLightning(object):
    """
   Randomly transforms image color using fixed PCA over ImageNet.

   The degree of color jittering is randomly sampled via a normal distribution,
   with standard deviation given by the scale parameter.
   """

    def __init__(self, scale):
        """
        Args:
            intensity_min (float): Minimum augmentation
            intensity_max (float): Maximum augmentation
        """

        self.eigen_vecs = np.array(
            [
                [-0.5675, 0.7192, 0.4009],
                [-0.5808, -0.0045, -0.814],
                [-0.5836, -0.6948, 0.4203]
            ]
        )
        self.eigen_vals = np.array([0.2175, 0.0188, 0.0045])

        self.scale = scale

    def __call__(self, sample):
        img = sample['img'].numpy()

        assert img.shape[-1] == 3, "Saturation only works on RGB images"

        weights = np.random.normal(scale=self.scale, size=3)

        src_weight = 1.0
        dst_weight = 1.0
        src_image = self.eigen_vecs.dot(weights * self.eigen_vals)

        if img.dtype == np.uint8:
            img = img.astype(np.float32)
            img = src_weight * src_image + dst_weight * img
            img = np.clip(img, 0, 255).astype(np.uint8)
        else:
            img = src_weight * src_image + dst_weight * img

        sample['img'] = torch.tensor(img)

        return sample
