from __future__ import division as _division
from __future__ import print_function as _print_function

import matplotlib.pyplot as plt
import numpy as np

import os as _os
import os.path as _path
import numpy as np
import cv2 as _cv2
from PIL import ImageFont
import numpy as _np
from hashlib import md5 as _md5

"""
    Modded the bounding box package to allow filled rectangles
"""

_LOC = _path.realpath(_path.join(_os.getcwd(), _path.dirname(__file__)))

# https://clrs.cc/
_COLOR_NAME_TO_RGB = dict(
    navy=((0, 38, 63), (119, 193, 250)),
    blue=((0, 120, 210), (173, 220, 252)),
    aqua=((115, 221, 252), (0, 76, 100)),
    teal=((15, 205, 202), (0, 0, 0)),
    olive=((52, 153, 114), (25, 58, 45)),
    green=((0, 204, 84), (15, 64, 31)),
    lime=((1, 255, 127), (0, 102, 53)),
    yellow=((255, 216, 70), (103, 87, 28)),
    orange=((255, 125, 57), (104, 48, 19)),
    red=((255, 47, 65), (131, 0, 17)),
    maroon=((135, 13, 75), (239, 117, 173)),
    fuchsia=((246, 0, 184), (103, 0, 78)),
    purple=((179, 17, 193), (241, 167, 244)),
    black=((24, 24, 24), (220, 220, 220)),
    gray=((168, 168, 168), (0, 0, 0)),
    silver=((220, 220, 220), (0, 0, 0))
)

_COLOR_NAMES = list(_COLOR_NAME_TO_RGB)

_DEFAULT_COLOR_NAME = "green"

_FONT_PATH = _os.path.join(_LOC, "Ubuntu-B.ttf")
_FONT_HEIGHT = 15
_FONT = ImageFont.truetype(_FONT_PATH, _FONT_HEIGHT)


def _rgb_to_bgr(color):
    return list(reversed(color))


def _color_image(image, font_color, background_color):
    return background_color + (font_color - background_color) * image / 255


def _get_label_image(text, font_color_tuple_bgr, background_color_tuple_bgr):
    text_image = _FONT.getmask(text)
    shape = list(reversed(text_image.size))
    bw_image = np.array(text_image).reshape(shape)

    image = [
        _color_image(bw_image, font_color, background_color)[None, ...]
        for font_color, background_color
        in zip(font_color_tuple_bgr, background_color_tuple_bgr)
    ]

    return np.concatenate(image).transpose(1, 2, 0)


def _add(image, left, top, right, bottom, label=None, color=None, filled=False):
    if type(image) is not _np.ndarray:
        raise TypeError("'image' parameter must be a numpy.ndarray")
    try:
        left, top, right, bottom = int(left), int(top), int(right), int(bottom)
    except ValueError:
        raise TypeError("'left', 'top', 'right' & 'bottom' must be a number")

    if label and type(label) is not str:
        raise TypeError("'label' must be a str")

    if label and not color:
        hex_digest = _md5(label.encode()).hexdigest()
        color_index = int(hex_digest, 16) % len(_COLOR_NAME_TO_RGB)
        color = _COLOR_NAMES[color_index]

    if not color:
        color = _DEFAULT_COLOR_NAME

    if type(color) is not str:
        raise TypeError("'color' must be a str")

    if color not in _COLOR_NAME_TO_RGB:
        msg = "'color' must be one of " + ", ".join(_COLOR_NAME_TO_RGB)
        raise ValueError(msg)

    colors = _COLOR_NAME_TO_RGB[color]# [_rgb_to_bgr(item) for item in _COLOR_NAME_TO_RGB[color]]
    color, color_text = colors

    if filled:
        _img = image.copy()
        line_thickness = -1
    else:
        _img = image
        line_thickness = 2

    _cv2.rectangle(_img, (left, top), (right, bottom), color, line_thickness)

    if label:
        _, image_width, _ = _img.shape

        label_image = _get_label_image(label, color_text, color)
        label_height, label_width, _ = label_image.shape

        rectangle_height, rectangle_width = 1 + label_height, 1 + label_width

        rectangle_bottom = top
        rectangle_left = max(0, min(left - 1, image_width - rectangle_width))

        rectangle_top = rectangle_bottom - rectangle_height
        rectangle_right = rectangle_left + rectangle_width

        label_top = rectangle_top + 1

        if rectangle_top < 0:
            rectangle_top = top
            rectangle_bottom = rectangle_top + label_height + 1

            label_top = rectangle_top

        label_left = rectangle_left + 1
        label_bottom = label_top + label_height
        label_right = label_left + label_width

        rec_left_top = (rectangle_left, rectangle_top)
        rec_right_bottom = (rectangle_right, rectangle_bottom)

        _cv2.rectangle(_img, rec_left_top, rec_right_bottom, color, -1)

        _img[label_top:label_bottom, label_left:label_right, :] = label_image

    if filled:
        alpha = 0.4
        _img = _cv2.addWeighted(_img, alpha, image, (1 - alpha), 0)

    return _img


def add_bbox_xywh(img, x, y, w, h, label=None):
    img = to_numpy_array(img)
    img = _add(img, x, y, x + w, y + h, label)
    return img


def add_bbox_xyxy(img, x, y, x_2, y_2, label=None, color=None):
    img = to_numpy_array(img)
    img = _add(img, x, y, x_2, y_2, label, color=color)
    return img


def add_filled_bbox_xyxy(img, x, y, x_2, y_2, label=None, color=None):
    img = to_numpy_array(img)
    img = _add(img, x, y, x_2, y_2, label, color=color, filled=True)
    return img


def box_cxcywh_to_xyxy(x_c, y_c, w, h):
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return b


def add_bbox_cxcyxy_ratio(img, c_x_r, c_y_r, x_2_r, y_2_r, label=None):
    img = to_numpy_array(img)
    x_c = img.shape[1] * c_x_r
    w = img.shape[1] * x_2_r
    y_c = img.shape[0] * c_y_r
    h = img.shape[0] * y_2_r

    x, y, x_2, y_2 = box_cxcywh_to_xyxy(x_c, y_c, w, h)
    x, y = max(x, 0), max(y, 0)

    img = _add(img, x, y, x_2, y_2, label)
    return img


def show_img(img):
    plt.figure()
    plt.imshow(img)
    plt.show()


def to_numpy_array(img):
    if not isinstance(img, np.ndarray):
        img = img.numpy()
        if len(img.shape) == 3 and img.shape[0] == 3:
            img = img.transpose(1, 2, 0)

        img = np.uint8((img - img.min()) / (img.max() - img.min()) * 255)
        img = np.ascontiguousarray(img)
    return img
