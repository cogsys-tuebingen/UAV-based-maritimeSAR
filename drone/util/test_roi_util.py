import unittest
from unittest import TestCase

import numpy as np
from typing import List

from util.roi_util import ROISplitter, Axis, ROISeperator


class TestROISplitter(TestCase):
    # Testing with np arrays to have a better visual result in case of printing

    def test_split_rois_no_splits_axis_0(self):
        split_size = 4
        splitter = ROISplitter(split_size, Axis.X_AXIS)
        image = np.zeros((8, 8))
        rois = [np.array([0, 1, 4, 1]), np.array([4, 2, 3, 2])]
        image = self.rois_to_ones(image, rois)
        splitted_image = (image[:split_size, :], image[split_size:, :])  # = expected result
        splitted_rois = splitter.split_rois(rois)
        splitted_images_zeros = (
            np.zeros((split_size, image.shape[1])), np.zeros((image.shape[0] - split_size, image.shape[1])))
        result_1 = self.rois_to_ones(splitted_images_zeros[0], splitted_rois[0])
        result_2 = self.rois_to_ones(splitted_images_zeros[1], splitted_rois[1])
        np.testing.assert_equal(splitted_image, (result_1, result_2))

    def test_split_rois_with_splits_axis_0(self):
        split_size = 4
        splitter = ROISplitter(split_size, Axis.X_AXIS)
        image = np.zeros((8, 8))
        rois = [np.array([2, 1, 4, 1]), np.array([4, 2, 3, 2])]
        image = self.rois_to_ones(image, rois)
        splitted_image = (image[:split_size, :], image[split_size:, :])  # = expected result
        splitted_rois = splitter.split_rois(rois)
        splitted_images_zeros = (
            np.zeros((split_size, image.shape[1])), np.zeros((image.shape[0] - split_size, image.shape[1])))
        result_1 = self.rois_to_ones(splitted_images_zeros[0], splitted_rois[0])
        result_2 = self.rois_to_ones(splitted_images_zeros[1], splitted_rois[1])
        np.testing.assert_equal(splitted_image, (result_1, result_2))

    def test_split_rois_with_splits_axis_0_split_size_2(self):
        split_size = 2
        splitter = ROISplitter(split_size, Axis.X_AXIS)
        image = np.zeros((8, 8))
        rois = [np.array([2, 1, 4, 1]), np.array([4, 2, 3, 2])]
        image = self.rois_to_ones(image, rois)
        splitted_image = (image[:split_size, :], image[split_size:, :])  # = expected result
        splitted_rois = splitter.split_rois(rois)
        splitted_images_zeros = (
            np.zeros((split_size, image.shape[1])), np.zeros((image.shape[0] - split_size, image.shape[1])))
        result_1 = self.rois_to_ones(splitted_images_zeros[0], splitted_rois[0])
        result_2 = self.rois_to_ones(splitted_images_zeros[1], splitted_rois[1])
        np.testing.assert_equal(splitted_image, (result_1, result_2))

    def test_split_rois_no_splits_axis_1(self):
        split_size = 4
        splitter = ROISplitter(split_size, Axis.Y_AXIS)
        image = np.zeros((8, 8))
        rois = [np.array([2, 0, 1, 4]), np.array([2, 4, 2, 3])]
        image = self.rois_to_ones(image, rois)
        splitted_image = (image[:, :split_size], image[:, split_size:])  # = expected result
        splitted_rois = splitter.split_rois(rois)
        splitted_images_zeros = (
            np.zeros((image.shape[0], split_size)), np.zeros((image.shape[0], image.shape[1] - split_size)))
        result_1 = self.rois_to_ones(splitted_images_zeros[0], splitted_rois[0])
        result_2 = self.rois_to_ones(splitted_images_zeros[1], splitted_rois[1])
        np.testing.assert_equal(splitted_image, (result_1, result_2))

    def test_split_rois_with_splits_axis_1(self):
        split_size = 4
        splitter = ROISplitter(split_size, Axis.Y_AXIS)
        image = np.zeros((8, 8))
        rois = [np.array([2, 1, 4, 1]), np.array([4, 2, 3, 2])]
        image = self.rois_to_ones(image, rois)
        splitted_image = (image[:, :split_size], image[:, split_size:])  # = expected result
        splitted_rois = splitter.split_rois(rois)
        splitted_images_zeros = (
            np.zeros((image.shape[0], split_size)), np.zeros((image.shape[0], image.shape[1] - split_size)))
        result_1 = self.rois_to_ones(splitted_images_zeros[0], splitted_rois[0])
        result_2 = self.rois_to_ones(splitted_images_zeros[1], splitted_rois[1])
        np.testing.assert_equal(splitted_image, (result_1, result_2))

    def test_split_rois_with_splits_axis_1_split_size_6(self):
        split_size = 6
        splitter = ROISplitter(split_size, Axis.Y_AXIS)
        image = np.zeros((8, 8))
        rois = [np.array([2, 1, 4, 1]), np.array([4, 2, 3, 2])]
        image = self.rois_to_ones(image, rois)
        splitted_image = (image[:, :split_size], image[:, split_size:])  # = expected result
        splitted_rois = splitter.split_rois(rois)
        splitted_images_zeros = (
            np.zeros((image.shape[0], split_size)), np.zeros((image.shape[0], image.shape[1] - split_size)))
        result_1 = self.rois_to_ones(splitted_images_zeros[0], splitted_rois[0])
        result_2 = self.rois_to_ones(splitted_images_zeros[1], splitted_rois[1])
        np.testing.assert_equal(splitted_image, (result_1, result_2))

    def rois_to_ones(self, image: np.ndarray, rois: List[np.ndarray]):
        for roi in rois:
            image[roi[0]:roi[0] + roi[2], roi[1]:roi[1] + roi[3]] = 1
        return image


class TestROISeperator(TestCase):
    def test_combine_overlapping_rois_empty(self):
        rois = []
        expected_result = []
        result = ROISeperator.unify_overlapping_rois(rois)
        np.testing.assert_equal(result, expected_result)

    def test_combine_overlapping_rois_not_overlapping(self):
        rois = [np.array([0, 0, 1, 1]), np.array([1, 1, 2, 2])]
        expected_result = [np.array([0, 0, 1, 1]), np.array([1, 1, 2, 2])]
        result = ROISeperator.unify_overlapping_rois(rois)
        np.testing.assert_equal(result, expected_result)

    def test_combine_overlapping_rois_overlapping(self):
        rois = [np.array([1, 1, 2, 2]), np.array([2, 2, 2, 2])]
        expected_result = [np.array([1, 1, 3, 3])]
        result = ROISeperator.unify_overlapping_rois(rois)
        np.testing.assert_equal(result, expected_result)

    def test_combine_overlapping_rois_overlapping_switched(self):
        rois = [np.array([1, 1, 2, 2]), np.array([2, 2, 2, 2])]
        expected_result = [np.array([1, 1, 3, 3])]
        result = ROISeperator.unify_overlapping_rois(rois)
        np.testing.assert_equal(result, expected_result)

    def test_combine_overlapping_rois_totally_overlapping(self):
        rois = [np.array([1, 1, 4, 4]), np.array([2, 2, 1, 1])]
        expected_result = [np.array([1, 1, 4, 4])]
        result = ROISeperator.unify_overlapping_rois(rois)
        np.testing.assert_equal(result, expected_result)

    def test_combine_overlapping_rois_totally_overlapping_switched(self):
        rois = [np.array([2, 2, 1, 1]), np.array([1, 1, 4, 4])]
        expected_result = [np.array([1, 1, 4, 4])]
        result = ROISeperator.unify_overlapping_rois(rois)
        np.testing.assert_equal(result, expected_result)

    def test_combine_overlapping_rois_overlapping_x(self):
        rois = [np.array([0, 0, 2, 2]), np.array([1, 0, 2, 2])]
        expected_result = [np.array([0, 0, 3, 2])]
        result = ROISeperator.unify_overlapping_rois(rois)
        np.testing.assert_equal(result, expected_result)

    def test_combine_overlapping_rois_overlapping_x_switched(self):
        rois = [np.array([1, 0, 2, 2]), np.array([0, 0, 2, 2])]
        expected_result = [np.array([0, 0, 3, 2])]
        result = ROISeperator.unify_overlapping_rois(rois)
        np.testing.assert_equal(result, expected_result)

    def test_combine_overlapping_rois_overlapping_x_2(self):
        rois = [np.array([1, 1, 2, 2]), np.array([2, 0, 2, 2])]
        expected_result = [np.array([1, 0, 3, 3])]
        result = ROISeperator.unify_overlapping_rois(rois)
        np.testing.assert_equal(result, expected_result)

    def test_combine_overlapping_rois_overlapping_x_2_switched(self):
        rois = [np.array([2, 0, 2, 2]), np.array([1, 1, 2, 2])]
        expected_result = [np.array([1, 0, 3, 3])]
        result = ROISeperator.unify_overlapping_rois(rois)
        np.testing.assert_equal(result, expected_result)

    def test_combine_overlapping_rois_overlapping_y(self):
        rois = [np.array([0, 0, 2, 2]), np.array([0, 1, 2, 2])]
        expected_result = [np.array([0, 0, 2, 3])]
        result = ROISeperator.unify_overlapping_rois(rois)
        np.testing.assert_equal(result, expected_result)

    def test_combine_overlapping_rois_overlapping_y_switched(self):
        rois = [np.array([0, 1, 2, 2]), np.array([0, 0, 2, 2])]
        expected_result = [np.array([0, 0, 2, 3])]
        result = ROISeperator.unify_overlapping_rois(rois)
        np.testing.assert_equal(result, expected_result)

    def test_combine_overlapping_rois_multiple_overlapping(self):
        rois = [np.array([0, 0, 2, 2]), np.array([0, 1, 2, 3]), np.array([1, 3, 2, 2])]
        expected_result = [np.array([0, 0, 3, 5])]
        result = ROISeperator.unify_overlapping_rois(rois)
        np.testing.assert_equal(result, expected_result)

    def test_combine_overlapping_rois_multiple_overlapping_switched(self):
        rois = [np.array([1, 3, 2, 2]), np.array([0, 0, 2, 2]), np.array([0, 1, 2, 3])]
        expected_result = [np.array([0, 0, 3, 5])]
        result = ROISeperator.unify_overlapping_rois(rois)
        np.testing.assert_equal(result, expected_result)

    def test_combine_overlapping_rois_multiple(self):
        rois = [np.array([1, 1, 2, 2]), np.array([2, 2, 2, 2]), np.array([5, 5, 2, 2]),
                np.array([6, 2, 2, 1]), np.array([7, 1, 3, 3])]
        expected_result = [np.array([1, 1, 3, 3]), np.array([5, 5, 2, 2]),
                           np.array([6, 1, 4, 3])]
        result = ROISeperator.unify_overlapping_rois(rois)
        np.testing.assert_equal(result, expected_result)

    def test_combine_overlapping_rois_multiple_2(self):
        rois = [np.array([1, 1, 2, 2]), np.array([2, 2, 2, 2]), np.array([5, 5, 2, 2]),
                np.array([6, 2, 2, 1]), np.array([7, 1, 3, 3]), np.array([6, 4, 2, 2])]
        expected_result = [np.array([1, 1, 3, 3]), np.array([6, 1, 4, 3]), np.array([5, 4, 3, 3])]
        result = ROISeperator.unify_overlapping_rois(rois)
        np.testing.assert_equal(result, expected_result)

    def test_separate_overlapping_rois_one_axis_same_y(self):
        rois = [np.array([0, 0, 2, 2]), np.array([1, 0, 2, 2])]
        expected_result = [np.array([0, 0, 3, 2])]
        result = ROISeperator.separate_overlapping_rois(rois)
        np.testing.assert_equal(result, expected_result)

    def test_separate_overlapping_rois_one_axis_same_x(self):
        rois = [np.array([0, 0, 2, 2]), np.array([0, 1, 2, 2])]
        expected_result = [np.array([0, 0, 2, 3])]
        result = ROISeperator.separate_overlapping_rois(rois)
        np.testing.assert_equal(result, expected_result)

    def test_separate_overlapping_rois_one_axis_same_x_switched(self):
        rois = [np.array([0, 1, 2, 2]), np.array([0, 0, 2, 2])]
        expected_result = [np.array([0, 0, 2, 3])]
        result = ROISeperator.separate_overlapping_rois(rois)
        np.testing.assert_equal(result, expected_result)

    def test_separate_overlapping_rois_overlapping_two_x_y_2(self):
        rois = [np.array([1, 1, 2, 2]), np.array([0, 0, 2, 2])]
        expected_result = [np.array([2, 1, 1, 1]), np.array([1, 2, 2, 1]), np.array([0, 0, 2, 2])]
        result = ROISeperator.separate_overlapping_rois(rois)
        np.testing.assert_equal(result, expected_result)

    def test_separate_overlapping_rois_overlapping_two_x_y_3(self):
        rois = [np.array([2, 1, 3, 3]), np.array([0, 0, 3, 3])]
        expected_result = [np.array([3, 1, 2, 2]), np.array([2, 3, 3, 1]), np.array([0, 0, 3, 3])]
        result = ROISeperator.separate_overlapping_rois(rois)
        np.testing.assert_equal(result, expected_result)

    def test_separate_overlapping_rois_overlapping_two_x_y_4(self):
        rois = [np.array([1, 2, 3, 3]), np.array([0, 0, 3, 3])]
        expected_result = [np.array([3, 2, 1, 1]), np.array([1, 3, 3, 2]), np.array([0, 0, 3, 3])]
        result = ROISeperator.separate_overlapping_rois(rois)
        np.testing.assert_equal(result, expected_result)

    def test_separate_overlapping_rois_overlapping_two_x_y_5(self):
        rois = [np.array([3, 3, 5, 5]), np.array([1, 1, 5, 5])]
        expected_result = [np.array([6, 3, 2, 3]), np.array([3, 6, 5, 2]), np.array([1, 1, 5, 5])]
        result = ROISeperator.separate_overlapping_rois(rois)
        np.testing.assert_equal(result, expected_result)

    def test_separate_overlapping_rois_overlapping_two_x_y_6(self):
        rois = [np.array([2, 0, 2, 2]), np.array([0, 1, 3, 3])]
        expected_result = [np.array([2, 0, 2, 1]), np.array([3, 1, 1, 1]), np.array([0, 1, 3, 3])]
        result = ROISeperator.separate_overlapping_rois(rois)
        np.testing.assert_equal(result, expected_result)

    def test_separate_overlapping_rois_overlapping_two_x_y_7(self):
        rois = [np.array([3, 1, 3, 3]), np.array([1, 3, 3, 3])]
        expected_result = [np.array([3, 1, 3, 2]), np.array([4, 3, 2, 1]), np.array([1, 3, 3, 3])]
        result = ROISeperator.separate_overlapping_rois(rois)
        np.testing.assert_equal(result, expected_result)

    def test_separate_overlapping_rois_overlapping_two_x_y_2_switched(self):
        rois = [np.array([0, 0, 2, 2]), np.array([1, 1, 2, 2])]
        expected_result = [np.array([0, 0, 2, 1]), np.array([0, 1, 1, 1]), np.array([1, 1, 2, 2])]
        result = ROISeperator.separate_overlapping_rois(rois)
        np.testing.assert_equal(result, expected_result)

    def test_separate_overlapping_rois_overlapping_two_x_y_3_switched(self):
        rois = [np.array([0, 0, 3, 3]), np.array([2, 1, 3, 3])]
        expected_result = [np.array([0, 0, 3, 1]), np.array([0, 1, 2, 2]), np.array([2, 1, 3, 3])]
        result = ROISeperator.separate_overlapping_rois(rois)
        np.testing.assert_equal(result, expected_result)

    def test_separate_overlapping_rois_overlapping_two_x_y_4_switched(self):
        rois = [np.array([0, 0, 3, 3]), np.array([1, 2, 3, 3])]
        expected_result = [np.array([0, 0, 3, 2]), np.array([0, 2, 1, 1]), np.array([1, 2, 3, 3])]
        result = ROISeperator.separate_overlapping_rois(rois)
        np.testing.assert_equal(result, expected_result)

    def test_separate_overlapping_rois_overlapping_two_x_y_5_switched(self):
        rois = [np.array([1, 1, 5, 5]), np.array([3, 3, 5, 5])]
        expected_result = [np.array([1, 1, 5, 2]), np.array([1, 3, 2, 3]), np.array([3, 3, 5, 5])]
        result = ROISeperator.separate_overlapping_rois(rois)
        np.testing.assert_equal(result, expected_result)

    def test_separate_overlapping_rois_overlapping_two_x_y_6_switched(self):
        rois = [np.array([0, 1, 3, 3]), np.array([2, 0, 2, 2])]
        expected_result = [np.array([2, 0, 2, 1]), np.array([3, 1, 1, 1]), np.array([0, 1, 3, 3])]
        result = ROISeperator.separate_overlapping_rois(rois)
        np.testing.assert_equal(result, expected_result)

    def test_separate_overlapping_rois_overlapping_two_x_y_7_switched(self):
        rois = [np.array([1, 3, 3, 3]), np.array([3, 1, 3, 3])]
        expected_result = [np.array([3, 1, 3, 2]), np.array([4, 3, 2, 1]), np.array([1, 3, 3, 3])]
        result = ROISeperator.separate_overlapping_rois(rois)
        np.testing.assert_equal(result, expected_result)

    def test_separate_overlapping_rois_same_x_different_height_1(self):
        rois = [np.array([4, 3, 3, 2]), np.array([1, 1, 5, 5])]
        expected_result = [np.array([6, 3, 1, 2]), np.array([1, 1, 5, 5])]
        result = ROISeperator.separate_overlapping_rois(rois)
        np.testing.assert_equal(result, expected_result)

    def test_separate_overlapping_rois_same_x_different_height_1_switched(self):
        rois = [np.array([1, 1, 5, 5]), np.array([4, 3, 3, 2])]
        expected_result = [np.array([6, 3, 1, 2]), np.array([1, 1, 5, 5])]
        result = ROISeperator.separate_overlapping_rois(rois)
        np.testing.assert_equal(result, expected_result)

    def test_separate_overlapping_rois_same_x_different_height_2(self):
        rois = [np.array([0, 0, 2, 3]), np.array([1, 0, 2, 2])]
        expected_result = [np.array([2, 0, 1, 2]), np.array([0, 0, 2, 3])]
        result = ROISeperator.separate_overlapping_rois(rois)
        np.testing.assert_equal(result, expected_result)

    def test_separate_overlapping_rois_same_x_different_height_2_switched(self):
        rois = [np.array([1, 0, 2, 2]), np.array([0, 0, 2, 3])]
        expected_result = [np.array([2, 0, 1, 2]), np.array([0, 0, 2, 3])]
        result = ROISeperator.separate_overlapping_rois(rois)
        np.testing.assert_equal(result, expected_result)

    def test_separate_overlapping_rois_same_y_different_width_1(self):
        rois = [np.array([3, 4, 2, 3]), np.array([1, 1, 5, 5])]
        expected_result = [np.array([3, 6, 2, 1]), np.array([1, 1, 5, 5])]
        result = ROISeperator.separate_overlapping_rois(rois)
        np.testing.assert_equal(result, expected_result)

    def test_separate_overlapping_rois_same_y_different_width_1_switched(self):
        rois = [np.array([3, 4, 2, 3]), np.array([1, 1, 5, 5])]
        expected_result = [np.array([3, 6, 2, 1]), np.array([1, 1, 5, 5])]
        result = ROISeperator.separate_overlapping_rois(rois)
        np.testing.assert_equal(result, expected_result)

    def test_separate_overlapping_rois_same_y_different_width_2(self):
        rois = [np.array([0, 0, 3, 2]), np.array([0, 1, 2, 2])]
        expected_result = [np.array([0, 2, 2, 1]), np.array([0, 0, 3, 2])]
        result = ROISeperator.separate_overlapping_rois(rois)
        np.testing.assert_equal(result, expected_result)

    def test_separate_overlapping_rois_same_y_different_width_2_switched(self):
        rois = [np.array([0, 0, 2, 3]), np.array([1, 0, 2, 2])]
        expected_result = [np.array([2, 0, 1, 2]), np.array([0, 0, 2, 3])]
        result = ROISeperator.separate_overlapping_rois(rois)
        np.testing.assert_equal(result, expected_result)

    def test_one_roi_in_another(self):
        rois = [np.array([1, 1, 5, 5]), np.array([1, 1, 2, 2])]
        expected_result = [np.array([1, 1, 5, 5])]
        result = ROISeperator.separate_overlapping_rois(rois)
        np.testing.assert_equal(result, expected_result)

    def test_one_roi_in_another_switched(self):
        rois = [np.array([1, 1, 2, 2]), np.array([1, 1, 5, 5])]
        expected_result = [np.array([1, 1, 5, 5])]
        result = ROISeperator.separate_overlapping_rois(rois)
        np.testing.assert_equal(result, expected_result)

    def test_separate_multiple_rois(self):
        rois = [np.array([1, 1, 3, 4]), np.array([3, 3, 3, 3]), np.array([5, 4, 3, 5]), np.array([7, 7, 2, 3])]
        expected_result = [np.array([4, 3, 2, 1]), np.array([1, 1, 3, 3]), np.array([1, 4, 2, 1]),
                           np.array([3, 4, 2, 2]), np.array([5, 4, 3, 3]), np.array([5, 7, 2, 2]),
                           np.array([7, 7, 2, 3])]
        result = ROISeperator.separate_overlapping_rois(rois)
        np.testing.assert_equal(result, expected_result)

    def test_separate_multiple_rois_2(self):
        rois = [np.array([0, 0, 2, 2]), np.array([1, 1, 2, 2]), np.array([2, 2, 2, 2]), np.array([2, 2, 3, 3])]
        expected_result = [np.array([2, 1, 1, 1]), np.array([0, 0, 2, 2]), np.array([1, 2, 1, 1]),
                           np.array([2, 2, 3, 3])]
        result = ROISeperator.separate_overlapping_rois(rois)
        np.testing.assert_equal(result, expected_result)

    def test_separate_multiple_rois_3(self):
        rois = [np.array([0, 0, 2, 2]), np.array([1, 0, 2, 2]), np.array([2, 0, 2, 2]), np.array([3, 0, 2, 2])]
        expected_result = [np.array([0, 0, 5, 2])]
        result = ROISeperator.separate_overlapping_rois(rois)
        np.testing.assert_equal(result, expected_result)

    def test_separate_multiple_rois_4(self):
        rois = [np.array([3, 0, 2, 2]), np.array([2, 0, 2, 2]), np.array([1, 0, 2, 2]), np.array([0, 0, 2, 2])]
        expected_result = [np.array([0, 0, 5, 2])]
        result = ROISeperator.separate_overlapping_rois(rois)
        np.testing.assert_equal(result, expected_result)
