import numpy as np


# @dataclass( order=False,frozen=False)
class DataFrame():

    def __init__(self, id, main_img: np.array, rois: [], roi_scores: [], custom_rois: []):
        self.id = id
        self.main_img = main_img
        self.rois = rois
        self.roi_scores = roi_scores
        self.custom_rois = custom_rois
