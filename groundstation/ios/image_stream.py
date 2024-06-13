import cv2
import time
import os
from typing import Callable

import main
from config import ARM_DEVICE_VIDEO_IP, USE_ARRIVAL_TIMESTAMP_FOR_SYNC, VIDEO_STREAM_DELAY, ARM_DEVICE_UDP_STREAM_PORT
from gui.app import App


def image_retrieve_thread_demo(root):
    import glob
    imgs = glob.glob('resources/test.jpg')
    idx = 0

    while root.running:
        img = cv2.imread(imgs[idx],
                         cv2.IMREAD_LOAD_GDAL | cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        root.set_img(img)
        time.sleep(1)

        idx += 1
        if idx >= len(imgs):
            idx = 0


def image_retrieve_thread_rtsp(root: App, submit_frame_func: Callable):
    # for vlc rtsp server it is necessary to use the udp protocol
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport\;udp"

    stream_url = f'rtsp://{ARM_DEVICE_VIDEO_IP}:8554/stream'
    stream = None
    frame_id = 0

    while root.running:
        try:
            while stream is None:
                stream = cv2.VideoCapture(stream_url)
                frame_id = 0

                if stream is None or not stream.isOpened():
                    stream = None
                    raise ConnectionError

            ret, frame = stream.read()

            frame_id += 1
            if USE_ARRIVAL_TIMESTAMP_FOR_SYNC:
                timestamp = (time.time_ns() // 1e7)
            else:
                timestamp = int(stream.get(cv2.CAP_PROP_POS_MSEC))
            root.update_state(image_error=False, image_connected=True)
            submit_frame_func(timestamp, frame)

        except Exception as e:
            stream = None
            print(f"! {e}")
            print("! Retry in a seconds..")
            root.update_state(image_error=True)
            time.sleep(1)


def image_retrieve_thread_udp(root: App, submit_frame_func: Callable, port: int):
    "FIMXE: still configured for the demo udp stream"
    stream_url = f'udp://{ARM_DEVICE_VIDEO_IP}:{port}?overrun_nonfatal=1&fifo_size=50000000'
    stream_url = f'udp://127.0.0.1:{port}?overrun_nonfatal=1&fifo_size=50000000'
    stream = None
    frame_id = 0

    while root.running:
        try:
            while stream is None:
                stream = cv2.VideoCapture(stream_url, cv2.CAP_FFMPEG)
                frame_id = 0

                if stream is None or not stream.isOpened():
                    stream = None
                    raise ConnectionError

            ret, frame = stream.read()

            frame_id += 1
            if USE_ARRIVAL_TIMESTAMP_FOR_SYNC:
                timestamp = (time.time_ns() // 1e7)
            else:
                timestamp = int(stream.get(cv2.CAP_PROP_POS_MSEC))
            #timestamp = 1000 * stream.get(cv2.CAP_PROP_POS_FRAMES)#int(stream.get(cv2.CAP_PROP_POS_MSEC)) # FIXME
            root.update_state(image_error=False, image_connected=True)
            submit_frame_func(timestamp, frame)

        except Exception as e:
            stream = None
            print(f"! {e}")
            print("! Retry in a seconds..")
            root.update_state(image_error=True)
            time.sleep(1)
