import os
import time
import cv2

import global_vars
from threading import Thread
import config


def image_load_loop(image_path:str,height, width):
   image_count = len(os.listdir(image_path))
   while global_vars.is_running:
        for current_frame in range(0,image_count):
                if not global_vars.is_running:
                        break
                frame = cv2.imread(os.path.join(image_path,f"frame{current_frame}.jpg"))
                frame = cv2.resize(frame, (width, height))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                global_vars.hdlr_lock.acquire(timeout=5)
                global_vars.last_cv_frame = frame
                global_vars.hdlr_lock.release()
                global_vars.handler_event.set()
                time.sleep(1)
