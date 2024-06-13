from typing import List
import numpy as np


class GroundStationStatusPacket:
    def __init__(self, timestamp, custom_rois: List[np.ndarray]):
        self.timestamp = timestamp
        self.custom_rois = custom_rois
