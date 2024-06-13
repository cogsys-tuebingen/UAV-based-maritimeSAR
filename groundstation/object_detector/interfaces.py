import numpy as np
import torch


class Detection:
    def __init__(self, x: int, y: int, width: int, height: int, class_id: int, confidence: float):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.class_id = class_id
        self.confidence = confidence

    def add_offset(self, off_x, off_y):
        self.x += off_x
        self.y += off_y
        return self

    def scale(self, factor):
        self.x = int(self.x * factor)
        self.y = int(self.y * factor)
        self.width = int(self.width * factor)
        self.height = int(self.height * factor)
        return self

    def clip_to(self, xmin, ymin, xmax, ymax):
        x1 = self.x
        y1 = self.y
        x2 = self.x + self.width
        y2 = self.y + self.height
        x1 = max(x1, xmin)
        y1 = max(y1, ymin)
        x2 = min(x2, xmax)
        y2 = min(y2, ymax)
        self.x = x1
        self.y = y1
        self.width = max(x2 - x1, 0)
        self.height = max(y2 - y1, 0)

    def is_empty(self):
        return self.width * self.height <= 0

    def to_numpy(self):
        return np.array([self.x, self.y, self.width, self.height])

    def __str__(self):
        return f'(x,y,w,h)={self.x},{self.y},{self.width},{self.height}, class_id={self.class_id}, confidence={self.confidence}'


class TrainedModelInterface:
    def __init__(self):
        super(TrainedModelInterface, self).__init__()

    def inference(self, roi_images: [np.array], confidence_threshold: float) -> [[Detection]]:
        raise NotImplementedError()

