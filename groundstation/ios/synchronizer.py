import copy

import time
from threading import Lock
from queue import Queue, Empty, Full
import numpy as np

from gui.app import App


class MetaDataAndVideoSynchronizer:
    def __init__(self, root:App, verbose):
        self.root = root

        self.frames_lock = Lock()
        self.frames = {}

        self.ready_frames = Queue(10)
        self.last_finished_timestamp = 0
        self.verbose = verbose

    def _prepare_frame(self, timestamp):
        self.frames.setdefault(timestamp, {'timestamp': timestamp, 'image': None, 'rois': None})

    def _finish_frame(self, timestamp):
        if self.frames[timestamp]['image'] is not None and self.frames[timestamp]['rois'] is not None:
            try:
                if self.ready_frames.full():
                    self.ready_frames.get_nowait()
                self.ready_frames.put_nowait(self.frames[timestamp])
            except Full:
                print("race condition")

            self.frames.pop(timestamp)

    def add_image_for_frame(self, timestamp, image):
        self.frames_lock.acquire()
        self._prepare_frame(timestamp)
        self.frames[timestamp]['image'] = image
        self._finish_frame(timestamp)
        self.frames_lock.release()

    def add_rois_for_frame(self, timestamp, rois: np.ndarray):
        self.frames_lock.acquire()
        self._prepare_frame(timestamp)
        self.frames[timestamp]['rois'] = rois
        self._finish_frame(timestamp)
        self.frames_lock.release()

    def exec(self):
        while self.root.running:

            if self.verbose:
                print(f"* SYNC_Unfinished packets: {len(self.frames.keys())}")
                print(f"* SYNC_Queue: {self.ready_frames.qsize()}")

            try:
                frame = self.ready_frames.get()
                timestamp = frame['timestamp']

                if timestamp > self.last_finished_timestamp:
                    self.root.set_img(copy.copy(frame['image']))
                    self.root.set_rois(copy.copy(frame['rois']))
                    self.last_finished_timestamp = timestamp
                    if self.verbose:
                        print(f"* SYNC_Finished frame: {timestamp}")
            except Empty:
                pass

            upcoming_timestamps = copy.copy(list(self.frames.keys()))
            remove_timestamps = [key for key in upcoming_timestamps if key < self.last_finished_timestamp]
            for k in remove_timestamps:
                self.frames.pop(k)


class VideoStreamsSynchronizer:
    def __init__(self, root:App, verbose):
        self.root = root

        self.frames_lock = Lock()
        self.frames = {}

        self.ready_frames = Queue(10)
        self.last_finished_timestamp = 0
        self.verbose = verbose

    def _prepare_frame(self, timestamp):
        self.frames.setdefault(timestamp, {'timestamp': timestamp, 'image_left': None, 'image_right': None})

    def _finish_frame(self, timestamp):
        if self.frames[timestamp]['image_left'] is not None and self.frames[timestamp]['image_right'] is not None:
            try:
                if self.ready_frames.full():
                    self.ready_frames.get_nowait()
                self.ready_frames.put_nowait(self.frames[timestamp])
            except Full:
                print("race condition")

            self.frames.pop(timestamp)

    def add_left_image_for_frame(self, timestamp, image):
        self.frames_lock.acquire()
        self._prepare_frame(timestamp)
        self.frames[timestamp]['image_left'] = image
        self._finish_frame(timestamp)
        self.frames_lock.release()

    def add_right_image_for_frame(self, timestamp, image):
        self.frames_lock.acquire()
        self._prepare_frame(timestamp)
        self.frames[timestamp]['image_right'] = image
        self._finish_frame(timestamp)
        self.frames_lock.release()

    def _combine_images(self, image_left, image_right):
        return np.concatenate((image_left, image_right), axis=1)

    def exec(self):
        while self.root.running:

            if self.verbose:
                print(f"* SYNC_Unfinished packets: {len(self.frames.keys())}")
                print(f"* SYNC_Queue: {self.ready_frames.qsize()}")

            try:
                frame = self.ready_frames.get()
                timestamp = frame['timestamp']

                if timestamp > self.last_finished_timestamp:
                    full_image = self._combine_images(frame['image_left'], frame['image_right'])
                    self.root.synchronizer.add_image_for_frame(timestamp, full_image)
                    self.last_finished_timestamp = timestamp
                    if self.verbose:
                        print(f"* SYNC_Finished frame: {timestamp}")
            except Empty:
                pass

            upcoming_timestamps = copy.copy(list(self.frames.keys()))
            remove_timestamps = [key for key in upcoming_timestamps if key < self.last_finished_timestamp]
            for k in remove_timestamps:
                self.frames.pop(k)
