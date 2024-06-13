import tkinter as tk
from object_detector.interfaces import Detection
import copy
from .util import AppState
import numpy as np
import config


class App(tk.Tk):
    def __init__(self, annotation_enabled_callback):
        super(App, self).__init__()

        self.annotation_enabled_callback = annotation_enabled_callback
        self.attributes('-zoomed', True)
        self.minsize(height=786, width=1024)
        self.title("Streams")
        # root.iconbitmap('resources/icon.ico')

        self.paused_input = False
        self.image = None
        self.rois = []
        self.detections = []

        self.custom_rois = []

        # ready for visualization
        self._ready_img = None
        self._ready_rois = np.array([])
        self._ready_detections = np.array([])

        self.class_dictionary = ClassDictionary(config.CLASSES)
        self.app_state = AppState.CONNECTING  # initial state
        # init internal states
        self._image_connected = False
        self._image_error = False
        self._meta_connected = False
        self._meta_error = False

    def set_img(self, img):
        if not self.paused_input:
            self.image = img

            if not config.SYNC_PREDICTIONS:
                self.set_processed_data(img=img)

    def get_img(self):
        return copy.deepcopy(self.image)

    def set_rois(self, rois: np.ndarray) -> None:
        """
               :param rois: expects xmin, ymin, width, height
               :return:
        """

        if not self.paused_input:
            self.rois = rois

            if not config.SYNC_PREDICTIONS:
                self.set_processed_data(rois=rois)

    def set_processed_data(self, img=None, rois=None, detections=None):
        """
            This method defines the img, rois and detections which are ready to be visualized
            Used to support the sync mode
        :param img:
        :param rois:
        :param detections:
        :return:
        """

        if img is not None:
            self._ready_img = img
        if rois is not None:
            self._ready_rois = rois
        if detections is not None:
            self._ready_detections = detections

    def get_ready_img(self):
        return self._ready_img

    def get_ready_rois(self) -> np.ndarray:
        return self._ready_rois

    def pause_input(self):
        self.paused_input = True

    def continue_input(self):
        self.paused_input = False

    def has_live_input(self):
        return not self.paused_input

    def add_custom_roi(self, custom_roi: np.ndarray):
        self.custom_rois.append(custom_roi)

    def get_custom_rois_copy(self) -> np.ndarray:
        return np.array(copy.deepcopy(self.custom_rois))

    def clear_custom_rois(self) -> None:
        self.custom_rois.clear()

    def get_rois(self):
        return copy.deepcopy(self.rois)

    def get_roi_region_imgs(self):
        """
        :return: the image parts of the rois
        """
        roi_regions = []
        roi_offsets = []
        if self.image is not None and len(self.rois) > 0:
            img_width, img_height = self.image.shape[1], self.image.shape[0]
            for roi in copy.deepcopy(self.rois):
                xmin, ymin, width, height = roi
                xmin, ymin = max(0, xmin), max(0, ymin)
                width = min(width, img_width-xmin)
                height = min(height, img_height-ymin)
                roi_region = self.image[ymin:(ymin + height), xmin:(xmin + width)]
                if roi_region.shape[0] * roi_region.shape[1] > 0:
                    roi_regions.append(roi_region)
                    roi_offsets.append((xmin, ymin, xmin + width, ymin + height))

        return roi_regions, roi_offsets

    def get_ready_detections(self) -> [Detection]:
        return copy.deepcopy(self._ready_detections)

    def get_detection_confidence_threshold(self) -> float:
        return config.CERTAINTY_THRESHOLD

    def get_state(self) -> AppState:
        return self.app_state

    def update_state(self, meta_error: bool = None, meta_connected: bool = None,
                     image_error: bool = None, image_connected: bool = None) -> None:
        if meta_error is not None:
            self._meta_error = meta_error
        if meta_connected is not None:
            self._meta_connected = meta_connected
        if image_error is not None:
            self._image_error = image_error
        if image_connected is not None:
            self._image_connected = image_connected
        if self._meta_error or self._image_error:
            self.app_state = AppState.ERROR
        elif self._image_connected and self._meta_connected:
            self.app_state = AppState.CONNECTED
        else:
            self.app_state = AppState.CONNECTING


# Command for generation
# CLASS_COLORS.append((random.randint(60, 255), random.randint(70, 255), random.randint(100, 255)))
CLASS_COLORS = [(200, 191, 160), (147, 230, 166), (240, 89, 208), (115, 132, 232), (169, 149, 176), (132, 176, 142),
                (145, 121, 197), (226, 214, 146), (225, 245, 212), (148, 117, 146), (210, 177, 186), (78, 107, 180),
                (228, 195, 195), (154, 211, 143), (200, 182, 242), (190, 169, 248), (85, 72, 237), (100, 116, 178),
                (146, 172, 148), (236, 112, 116), (135, 242, 221), (80, 197, 214), (168, 130, 163), (121, 243, 122),
                (230, 111, 125), (63, 187, 202), (114, 123, 128), (162, 166, 184), (108, 236, 179), (73, 84, 156),
                (111, 220, 249), (216, 172, 187), (185, 230, 255), (88, 132, 122), (67, 95, 219), (213, 194, 235),
                (107, 89, 126), (226, 234, 169), (227, 94, 222), (217, 225, 116), (246, 131, 230), (201, 172, 203),
                (140, 161, 246), (102, 237, 226), (84, 148, 189), (229, 113, 197), (117, 97, 117), (235, 79, 176),
                (201, 244, 242), (195, 82, 233), (192, 204, 173), (138, 203, 188), (96, 182, 103), (133, 95, 106),
                (108, 172, 103), (94, 94, 154), (203, 225, 182), (191, 219, 135), (175, 113, 117), (228, 147, 192),
                (243, 112, 135), (65, 70, 234), (214, 120, 207), (254, 235, 131), (213, 134, 209), (70, 253, 249),
                (166, 73, 162), (228, 254, 220), (211, 117, 226), (144, 118, 251), (121, 84, 151), (128, 224, 246),
                (77, 201, 226), (223, 114, 137), (195, 115, 193), (137, 108, 191), (142, 87, 214), (99, 231, 203),
                (159, 116, 232), (79, 97, 231), (61, 254, 241), (187, 238, 170), (118, 146, 226), (110, 111, 217),
                (236, 93, 113), (191, 189, 162), (187, 217, 172), (225, 189, 119), (79, 88, 255), (156, 95, 221),
                (88, 145, 140), (165, 129, 202), (139, 248, 229), (67, 230, 217), (168, 235, 224), (148, 208, 227),
                (251, 243, 183), (168, 114, 187), (73, 188, 122), (112, 230, 235)]


class ClassDictionary:
    def __init__(self, classes):
        self.classes = classes

    def get_color_for(self, class_id: int):
        if class_id > len(CLASS_COLORS):
            return (0, 0, 0)
        else:
            return CLASS_COLORS[class_id]

    def get_label_for(self, class_id: int):
        res = list(filter(lambda c: c['id'] == class_id, self.classes))
        if len(res) > 0:
            return res[0]["name"]
        else:
            return 'unknown'

    def get_labels(self):
        return list(map(lambda c: c["name"], self.classes))

    def get_id_for(self, label):
        res = list(filter(lambda c: c['name'] == label, self.classes))
        if len(res) > 0:
            return res[0]["id"]
        else:
            print(f"ID for label {label} not found")
            return -1
