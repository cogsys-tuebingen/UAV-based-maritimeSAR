import random
from enum import Enum

import numpy as np
from typing import List


class Axis(Enum):
    X_AXIS = 0  # split parallel to x-Axis
    Y_AXIS = 1  # split parallel to y-Axis


class ROISplitter:
    def __init__(self, split_size: int, axis: Axis):
        """

        @param split_size: size of first half. e.g. first 500 pixels (index 0-499)
        @param axis: like numpy axis 0 for horizontal split, 1 for vertical cut
        """
        self.split_size = split_size
        self.axis = axis

    def split_rois(self, rois: List[np.ndarray]):
        resulting_rois = ([], [])
        for roi in rois:
            if self.axis == Axis.X_AXIS:
                if roi[0] + roi[2] <= self.split_size:
                    resulting_rois[0].append(roi)
                elif roi[0] >= self.split_size:
                    roi[0] -= self.split_size
                    resulting_rois[1].append(roi)
                else:  # ROI in both parts of image
                    roi_1 = np.array([roi[0], roi[1], self.split_size - roi[0], roi[3]])
                    roi_2 = np.array([0, roi[1], roi[2] - roi_1[2], roi[3]])
                    resulting_rois[0].append(roi_1)
                    resulting_rois[1].append(roi_2)
            else:
                if roi[1] + roi[3] <= self.split_size:
                    resulting_rois[0].append(roi)
                elif roi[1] >= self.split_size:
                    roi[1] -= self.split_size
                    resulting_rois[1].append(roi)
                else:  # ROI in both parts of image
                    roi_1 = np.array([roi[0], roi[1], roi[2], self.split_size - roi[3]])
                    roi_2 = np.array([roi[0], 0, roi[2], roi[3] - roi_1[3]])
                    resulting_rois[0].append(roi_1)
                    resulting_rois[1].append(roi_2)

        return resulting_rois


class ROISeperator:
    @staticmethod
    def unify_overlapping_rois(rois: List[np.ndarray]) -> List[np.ndarray]:
        if len(rois) == 0:
            return []
        else:
            current_roi = rois.pop()
            drop_indexes_set = set()
            for i, roi in enumerate(rois):
                if (current_roi[0] + current_roi[2] > roi[0] >= current_roi[0] or
                    roi[0] + roi[2] > current_roi[0] >= roi[0]) and \
                        (current_roi[1] + current_roi[3] > roi[1] >= current_roi[1] or
                         roi[1] + roi[3] > current_roi[1] >= roi[1]):
                    current_roi[2] = max(current_roi[2], roi[2], roi[0] - current_roi[0] + roi[2],
                                         current_roi[0] - roi[0] + current_roi[2])
                    current_roi[3] = max(current_roi[3], roi[1] - current_roi[1] + roi[3], roi[3],
                                         current_roi[1] - roi[1] + current_roi[3])
                    current_roi[0] = min(current_roi[0], roi[0])
                    current_roi[1] = min(current_roi[1], roi[1])
                    drop_indexes_set.add(i)
            for index in reversed(list(drop_indexes_set)):  # remove the highest first otherwise out of bounds exception
                rois.pop(index)
            if len(drop_indexes_set) == 0:
                return ROISeperator.unify_overlapping_rois(rois) + [current_roi]
            else:
                return ROISeperator.unify_overlapping_rois(
                    rois + [current_roi])  # add current roi again in case of multiple overlapping rois

    @staticmethod
    def separate_overlapping_rois(rois: List[np.ndarray]) -> List[np.ndarray]:
        # assuming roi[0],roi[1] describe top left and roi[2],roi[3] width, height
        if len(rois) <= 1:
            return rois
        else:
            current_roi = rois.pop()
            keep_current_roi = True
            resulting_rois = []
            new_rois = rois.copy()
            for i, roi in reversed(list(enumerate(rois))):  # reversed list needed for pop(i)
                # if rois are only overlap on one axis and have same height/width
                if (current_roi[0] == roi[0] and current_roi[2] == roi[2]) or \
                        (current_roi[1] == roi[1] and current_roi[3] == roi[3]):
                    current_roi[2] = max(current_roi[2], roi[2], roi[0] - current_roi[0] + roi[2],
                                         current_roi[0] - roi[0] + current_roi[2])
                    current_roi[3] = max(current_roi[3], roi[1] - current_roi[1] + roi[3], roi[3],
                                         current_roi[1] - roi[1] + current_roi[3])
                    current_roi[0] = min(current_roi[0], roi[0])
                    current_roi[1] = min(current_roi[1], roi[1])
                    new_rois.pop(i)
                    new_rois.append(current_roi)
                    keep_current_roi = False

                # one roi contains another
                elif current_roi[0] <= roi[0] <= (roi[0] + roi[2]) < current_roi[0] + current_roi[2] and \
                        current_roi[1] <= roi[1] <= (roi[1] + roi[3]) < current_roi[1] + current_roi[3]:
                    new_rois.pop(i)
                    new_rois.append(current_roi)
                    keep_current_roi = False
                elif roi[0] <= current_roi[0] <= (current_roi[0] + current_roi[2]) < roi[0] + roi[2] and \
                        roi[1] <= current_roi[1] <= (current_roi[1] + current_roi[3]) < roi[1] + roi[3]:
                    new_rois.append(roi)
                    new_rois.pop(i)
                    keep_current_roi = False
                # if rois overlap on one axis with different height/width
                # overlapping along y-Axis
                elif current_roi[1] <= roi[1] <= (roi[1] + roi[3]) <= current_roi[1] + current_roi[3] and \
                        current_roi[0] <= roi[0] < current_roi[0] + current_roi[2]:
                    new_roi = np.array(
                        [current_roi[0] + current_roi[2], roi[1],
                         (roi[0] + roi[2]) - (current_roi[0] + current_roi[2]), roi[3]])
                    new_rois.append(new_roi)
                    new_rois.append(current_roi)
                    new_rois.pop(i)
                    keep_current_roi = False
                elif roi[1] <= current_roi[1] <= (current_roi[1] + current_roi[3]) <= roi[1] + roi[3] and \
                        roi[0] <= current_roi[0] < roi[0] + roi[2]:
                    new_roi = np.array(
                        [roi[0] + roi[2], current_roi[1],
                         (current_roi[0] + current_roi[2]) - (roi[0] + roi[2]), current_roi[3]])
                    new_rois.append(new_roi)
                    new_rois.pop(i)
                    new_rois.append(roi)
                    keep_current_roi = False
                # ---- x-Axis
                elif current_roi[0] <= roi[0] <= (roi[0] + roi[2]) <= current_roi[0] + current_roi[2] and \
                        current_roi[1] <= roi[1] < current_roi[1] + current_roi[3]:
                    new_roi = np.array(
                        [roi[0], current_roi[1] + current_roi[3],
                         roi[2], (roi[1] + roi[3]) - (current_roi[1] + current_roi[3])])
                    new_rois.append(new_roi)
                    new_rois.pop(i)
                    new_rois.append(current_roi)
                    keep_current_roi = False

                elif roi[0] <= current_roi[0] <= (current_roi[0] + current_roi[2]) <= roi[0] + roi[2] and \
                        roi[1] <= current_roi[1] < roi[1] + roi[3]:
                    new_roi = np.array(
                        [current_roi[0], roi[1] + roi[3],
                         current_roi[2], (current_roi[1] + current_roi[3]) - (roi[1] + roi[3])])
                    new_rois.append(new_roi)
                    new_rois.pop(i)
                    resulting_rois.append(roi)
                    keep_current_roi = False

                # --------------------------------------------------
                # if roi overlaps other rois corner
                elif current_roi[0] <= roi[0] < current_roi[0] + current_roi[2] and \
                        current_roi[1] <= roi[1] < current_roi[1] + current_roi[3]:
                    new_roi_1 = np.array([current_roi[0] + current_roi[2], roi[1],
                                          (roi[0] + roi[2]) - (current_roi[0] + current_roi[2]),
                                          current_roi[1] + current_roi[3] - roi[1]])
                    new_roi_2 = np.array([roi[0], current_roi[1] + current_roi[3],
                                          roi[2], (roi[1] + roi[3]) - (current_roi[1] + current_roi[3])])
                    new_rois.append(new_roi_1)
                    new_rois.append(new_roi_2)
                    new_rois.pop(i)
                    new_rois.append(current_roi)
                    keep_current_roi = False
                elif current_roi[0] <= roi[0] < current_roi[0] + current_roi[2] and \
                        current_roi[1] < roi[1] + roi[3] < current_roi[1] + current_roi[3]:
                    new_roi_1 = np.array([roi[0], roi[1], roi[2],
                                          current_roi[1] - roi[1]])
                    new_roi_2 = np.array([current_roi[0] + current_roi[2], current_roi[1],
                                          (roi[0] + roi[2]) - (current_roi[0] + current_roi[2]),
                                          (roi[1] + roi[3]) - current_roi[1]])
                    new_rois.append(new_roi_1)
                    new_rois.append(new_roi_2)
                    new_rois.pop(i)
                    new_rois.append(current_roi)
                    keep_current_roi = False
                elif roi[0] <= current_roi[0] < roi[0] + roi[2] and \
                        roi[1] <= current_roi[1] < roi[1] + roi[3]:
                    new_roi_1 = np.array([roi[0], roi[1], roi[2],
                                          current_roi[1] - roi[1]])
                    new_roi_2 = np.array([roi[0], current_roi[1],
                                          current_roi[0] - roi[0], roi[3] - (current_roi[1] - roi[1])])
                    new_rois.append(new_roi_1)
                    new_rois.append(new_roi_2)
                    new_rois.pop(i)
                    new_rois.append(current_roi)
                    keep_current_roi = False
                elif roi[0] <= current_roi[0] < roi[0] + roi[2] and \
                        roi[1] < current_roi[1] + current_roi[3] < roi[1] + roi[3]:
                    new_roi_1 = np.array([current_roi[0], current_roi[1], current_roi[2],
                                          roi[1] - current_roi[1]])
                    new_roi_2 = np.array([roi[0] + roi[2], roi[1],
                                          (current_roi[0] + current_roi[2]) - (roi[0] + roi[2]),
                                          (current_roi[1] + current_roi[3]) - roi[1]])
                    new_rois.append(new_roi_1)
                    new_rois.append(new_roi_2)
                    new_rois.pop(i)
                    resulting_rois.append(roi)
                    keep_current_roi = False
            if keep_current_roi:
                resulting_rois.append(current_roi)

            return ROISeperator.separate_overlapping_rois(new_rois) + resulting_rois
