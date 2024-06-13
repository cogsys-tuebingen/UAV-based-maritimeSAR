import time
import cv2

import global_vars
from threading import Thread
import config


def capture_demo_video(heigth, width, verbose):
    cap = None

    while global_vars.is_running:
        cap = cv2.VideoCapture(config.DEMO_VIDEO_PATH)

        if not cap.isOpened():
            print("! Cannot open camera. Retry in a second")
            time.sleep(1)
            continue

        while global_vars.is_running:
            ret, frame = cap.read()
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break

            frame = cv2.resize(frame, (width, heigth))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            global_vars.hdlr_lock.acquire()
            global_vars.last_cv_frame = frame
            global_vars.hdlr_lock.release()
            global_vars.handler_event.set()

    if cap is not None and cap.isOpened():
        cap.release()


def start_demo_cam_thread(height, width, verbose):
    usbcam_thread = Thread(target=capture_demo_video, args=([int(height), int(width), verbose]))
    usbcam_thread.start()
    global_vars.thread_holder.append(usbcam_thread)
