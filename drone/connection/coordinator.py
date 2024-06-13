import queue
import time

import global_vars
from threading import Thread, Lock
import numpy as np
from queue import Queue
import copy
import cv2

import config
from .metadata_stream import meta_data_stream_loop
from .rtsp_stream import rtsp_stream_loop, STREAM_HEIGHT, STREAM_WIDTH, FRAME_DURATION, Gst
from data_structures.meta_data import MetaDataStructure


class StreamCoordinator:
    def __init__(self):
        self.roi_handler = Thread(target=meta_data_stream_loop, args=([self]))
        self.roi_handler.name = "roi-handler-thread"
        self.rtsp_handler = Thread(target=rtsp_stream_loop, args=([self]))
        self.rtsp_handler.name = "rtsp-handler-thread"

        self.frame_number_lock = Lock()
        self.frame_number = 0
        self.last_received_dataframe_id = 0

        self.next_frame_buffer_lock = Lock()
        self.next_frame_buffer = None

        self.next_meta_data_lock = Lock()
        self.next_meta_data = None
        self.meta_data_queue = Queue(10)

    def exec(self):
        self.roi_handler.start()
        global_vars.thread_holder.append(self.roi_handler)
        self.rtsp_handler.start()
        global_vars.thread_holder.append(self.rtsp_handler)

        pos = 0

        while global_vars.is_running:
            global_vars.new_dataframe_event.wait(timeout=1)
            global_vars.new_dataframe_event.clear()

            global_vars.det_lock.acquire()
            dataframe = copy.deepcopy(global_vars.last_data_frame)
            global_vars.last_data_frame = None
            global_vars.det_lock.release()

            if dataframe is None:
                continue

            self.next_meta_data_lock.acquire()
            # the frame id is set, when the stream requests the image
            rois = None
            if len(dataframe.custom_rois) > 0:
                rois = np.concatenate((dataframe.rois, dataframe.custom_rois), axis=0)
            else:
                rois = dataframe.rois
            self.next_meta_data = MetaDataStructure(None, None, rois)
            self.next_meta_data_lock.release()
            
            if config.COMPRESS_IRRELEVANT_REGIONS:
                dataframe.main_img = compress_irrelevant_regions(dataframe.main_img, rois)

            self.next_frame_buffer_lock.acquire()
            self.next_frame_buffer = dataframe.main_img
            self.next_frame_buffer_lock.release()
            pos += 1

    def get_next_frame(self) -> (np.array, int):
        self.frame_number_lock.acquire()
        if config.COORDINATE_TIMESTAMP_BASED:
            timestamp = int(time.time() * Gst.SECOND) - self.stream_init_timestamp_ms
        else:
            timestamp = self.frame_number * FRAME_DURATION
        self.frame_number_lock.release()

        self.next_frame_buffer_lock.acquire()
        frame = self.next_frame_buffer
        self.next_frame_buffer = None
        self.next_frame_buffer_lock.release()
        if frame is None:
            return (None, timestamp)

        self.next_meta_data_lock.acquire()
        meta_data = copy.copy(self.next_meta_data)
        # we want to keept the last meta data until we have new rois
        self.next_meta_data_lock.release()
        meta_data.update_frame_id(self.frame_number)
        meta_data.update_timestamp(timestamp)

        try:
            self.meta_data_queue.put_nowait(meta_data)
        except queue.Full:
            pass

        return (frame, timestamp)

    def reset_frame_number(self):
        self.frame_number_lock.acquire()
        self.frame_number = 0
        self.stream_init_timestamp_ms = int(time.time() * Gst.SECOND)
        self.frame_number_lock.release()

    def inc_frame_number(self):
        self.frame_number_lock.acquire()
        self.frame_number += 1
        self.frame_number_lock.release()

    def stop(self):
        global_vars.is_running = False


def compress_irrelevant_regions(img, rois):
    org_img = copy.copy(img)

    width, height, _ = img.shape

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (height // 8, width // 8))
    img = cv2.resize(img, (height, width))
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    for (x1, y1, width, height) in rois:
        img[y1:(y1+height), x1:(x1+width)] = org_img[y1:(y1+height), x1:(x1+width)]

    
    return img

