import numpy as np


class DataFrame:
    def __init__(self, main_img: np.array, main_shape: tuple(int, int), main_dtype: np.dtype,
                 scale: float, n_subs: int, sub_frames: list):
        self.main_img = main_img
        self.main_shape = main_shape
        self.main_dtype = main_dtype
        self.scale = scale
        self.n_subs = n_subs
        self.sub_frames = sub_frames


class SubFrame:
    def __init__(self, sub_img: np.array, sub_shape, sub_dtype: np.dtype,
                 sub_orig_coords: tuple(int, int, int, int)):
        self.sub_img = sub_img
        self.sub_shape = sub_shape
        self.sub_dtype = sub_dtype
        self.sub_orig_coords = sub_orig_coords  # xmin,ymin,xmax,ymax
