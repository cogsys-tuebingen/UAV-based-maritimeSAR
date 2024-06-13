from datetime import datetime
import os
import time
import numpy as np
import cv2
from vimba import *
from typing import Optional
import global_vars
from datetime import datetime

class MT_Handler:
    def __init__(self, target_height, target_width, mockup=False, save_path:str=None):
        self.target_height = target_height
        self.target_width = target_width
        self.mockup = mockup
        self.frame_counter = 0
        if save_path is not None:
            timestamp = str(datetime.now())
            self.save_path = os.path.join(save_path, timestamp)
            os.mkdir(self.save_path)
            print(f"#  Saving recorded images at {self.save_path}")
        else:
            self.save_path = None

    def __call__(self, cam: Camera, frame: Frame):
        if frame.get_status() == FrameStatus.Complete:
            cv_frame = np.frombuffer(frame._buffer, dtype=np.uint8).reshape(frame._frame.height, frame._frame.width)
            cv_frame = cv2.cvtColor(cv_frame, cv2.COLOR_BAYER_RG2RGB)
            cv_frame = cv2.cvtColor(cv_frame, cv2.COLOR_RGB2BGR)
            cv_frame = cv2.resize(cv_frame, (self.target_width, self.target_height))
            if not self.mockup:
                global_vars.hdlr_lock.acquire()
                global_vars.last_cv_frame = cv_frame.copy()
                global_vars.hdlr_lock.release()
                global_vars.handler_event.set()
            else:
                print("Received frame")
                cv2.imwrite("/tmp/frame.png", cv_frame)
            if self.save_path is not None:

                cv2.imwrite(os.path.join(self.save_path, f"frame{self.frame_counter}.jpg"), cv_frame)
                self.frame_counter = self.frame_counter  + 1

        cam.queue_frame(frame)


def get_camera(vimba, camera_id: Optional[str] = False) -> Camera:
    if camera_id:
        try:
            return vimba.get_camera_by_id(camera_id)
        except VimbaCameraError:
            raise Exception('Failed to access Camera \'{}\'. Abort.'.format(camera_id))

    else:
        cams = vimba.get_all_cameras()
        if not cams:
            raise Exception('No Cameras accessible. Abort.')
        return cams[0]


def setup_camera(cam: Camera, offset_x, offset_y, width, height):
    with cam:
        # Enable auto exposure time setting if camera supports it
        try:
            cam.Gain.set(20)
            cam.ExposureAuto.set('Continuous')
            cam.BalanceWhiteAuto.set('Once')
            # Query available, open_cv compatible pixel formats
            # prefer color formats over monochrome formats
            cam.set_pixel_format(PixelFormat.BayerRG8)
#            cam.GainAuto.set('Continuous')

            cam.Width.set(width)
            cam.Height.set(height)
            cam.OffsetX.set(offset_x)
            cam.OffsetY.set(offset_y)

        except (AttributeError, VimbaFeatureError):
            raise RuntimeError('Attribute error')

        # Enable white balancing if camera supports it
        try:
            cam.BalanceWhiteAuto.set('Continuous')

        except (AttributeError, VimbaFeatureError):
            pass

        # Try to adjust GeV packet size. This Feature is only available for GigE - Cameras.
        try:
            cam.GVSPAdjustPacketSize.run()

            while not cam.GVSPAdjustPacketSize.is_done():
                pass

        except (AttributeError, VimbaFeatureError):
            pass



def vimba_loop(offset_x, offset_y, width, height, mockup=False, save_path:str = None):
    with Vimba.get_instance() as vimba:
        with get_camera(vimba=vimba) as cam:
            # Start Streaming, wait for five seconds, stop streaming
            setup_camera(cam, offset_x, offset_y, 4112, 3008)
            handler = MT_Handler(height, width, mockup=mockup,save_path = save_path)
            try:
                # Start Streaming with a custom a buffer of 10 Frames (defaults to 5)
                cam.start_streaming(handler=handler, buffer_count=10)
                while global_vars.is_running:
                    time.sleep(1)
            except Exception as e:
                print(e)
            finally:
                cam.stop_streaming()
                

if __name__ == '__main__':
    offset_x, offset_y = 0, 0
    height, width = 1024, 1024
    mockup = True
    print("Start vimba")
    with Vimba.get_instance():
        print("got instance")
        with get_camera() as cam:
            print("got camera")

            # Start Streaming, wait for five seconds, stop streaming
            setup_camera(cam, offset_x, offset_y, 4112, 3008)
            print("camera was setup")
            handler = MT_Handler(height, width, mockup=mockup)
            print("handler created")

            try:
                # Start Streaming with a custom a buffer of 10 Frames (defaults to 5)
                cam.start_streaming(handler=handler, buffer_count=10)
                handler.shutdown_event.wait()

            finally:
                cam.stop_streaming()
                cam.close()
