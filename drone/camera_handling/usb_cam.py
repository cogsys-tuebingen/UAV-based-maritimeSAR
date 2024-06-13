import time
import cv2

import global_vars
from threading import Thread


def capture_usbcam(heigth, width, verbose):
    cap = None

    while global_vars.is_running:
        cap = cv2.VideoCapture(0)

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


def start_usb_cam_thread(height, width, verbose):
    usbcam_thread = Thread(target=capture_usbcam, args=([int(height), int(width), verbose]))
    usbcam_thread.start()
    global_vars.thread_holder.append(usbcam_thread)
