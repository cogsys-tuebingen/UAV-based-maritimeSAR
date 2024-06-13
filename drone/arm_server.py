import string
import numpy as np
import argparse
from util.data_frame import DataFrame
from threading import Thread
import time
import signal

from connection import control_stream_loop, \
    StreamCoordinator

import global_vars


def print_preamble():
    print('///////////////////////////////////////////////////////')
    print('/// ARM Software                                    ///')
    print('///////////////////////////////////////////////////////\n')


def parse_verbose(verbose):
    verbose_cam = True if verbose in ['cam', 'all'] else False
    verbose_detector = True if verbose in ['detector', 'all'] else False
    verbose_output = True if verbose in ['server', 'all'] else False
    verbose_rp = True if verbose in ['rp', 'all'] else False

    return verbose_cam, verbose_detector, verbose_output, verbose_rp


def header():
    print_preamble()
    parser = argparse.ArgumentParser(description='Asychronuous Grab openCV')
    parser.add_argument('--detector', help='TensorRT Checkpoint path')
    parser.add_argument('--image_size', default='1280x768',
                        help='image size of video stream')
    parser.add_argument('--detector_size',
                        help='image size of checkpoint model ex: 1280x640')
    parser.add_argument('--detection_frequency',
                        help='detection every X frames', type=int, required=True)
    parser.add_argument('--main_image_scale', default=0.5, type=float)
    parser.add_argument('--grayscale', default=True)
    parser.add_argument('--cam_type', default='vimba')
    parser.add_argument('--protocol', default='rtsp',
                        choices=['udp', 'tcp', 'rtsp'], help="The protocol for the video stream")
    parser.add_argument('--verbose', default='none')
    parser.add_argument('--compression', default='.jpg')
    parser.add_argument('--saliency', action='store_true')
    parser.add_argument('--best_p', default=0.1, type=float)
    parser.add_argument('--enlarge_bb', default=1.0, type=float)
    parser.add_argument('--dump_locally', default=False, action='store_true')
    parser.add_argument('--mockup_detection', default=False, action='store_true')
    parser.add_argument('--average_roi_frames', default=5, type=int)
    parser.add_argument('--image_save_path', help='Path to save images')
    parser.add_argument('--image_load_path', help='Path to load images')
    args = parser.parse_args()

    assert "x" in args.image_size
    args.w, args.h = args.image_size.split('x')
    args.w, args.h = int(args.w), int(args.h)

    assert "x" in args.detector_size
    d_w, d_h = args.detector_size.split('x')
    args.detector_size = (int(d_w), int(d_h))

    verbose_cam, verbose_detector, verbose_output, verbose_rp = parse_verbose(
        args.verbose)

    args.verbose = {
        'cam': verbose_cam,
        'detector': verbose_detector,
        'output': verbose_output,
        'rp': verbose_rp
    }

    return args


def init(args):
    # init the last holders
    global_vars.last_cv_frame = None 
    global_vars.last_data_frame = None

def _start_thread(target: callable, name: string, args:list=None, add_to_thread_holder=True)->Thread:
    """
    Starts a thread. Names it and adds it the global_vars.thread_holder
    """
    thread = Thread(target=target, args=args, )
    thread.name = name
    if add_to_thread_holder:
        global_vars.thread_holder.append(thread)
    thread.start()
    print(f"# Starting {name}")
    return thread

def main():
    args = header()
    init(args)
    print(f'Used camera type: {args.cam_type}')

    print("Start..")

    if not args.mockup_detection:
        from detection.detection_framework import detection_loop
        detection_thread = _start_thread(target=detection_loop,
                                name="Detection-Thread",
                                args=(
                                    [args.detector, args.detection_frequency, args.detector_size,
                                    args.verbose['detector'], args.saliency,
                                    args.best_p, args.enlarge_bb, 0.6, args.average_roi_frames]),
                                add_to_thread_holder=False)
    else:
        from detection.detection_framework import detection_mockup
        detection_thread = _start_thread(target=detection_mockup,
                                name = "Detection-Thread",
                                args=([]),
                                add_to_thread_holder=False)


    control_thread = _start_thread(target=control_stream_loop,name = "Control-Thread", args=([False]))

    coordinator = StreamCoordinator()
    coordinator_thread = _start_thread(target=coordinator.exec,name = "Coordinator-Thread", args=([]))

    if args.cam_type == 'usb':
        print('Using default USB-Cam')
        from camera_handling.usb_cam import start_usb_cam_thread
        start_usb_cam_thread(args.h, args.w, args.verbose['cam'])
    elif args.cam_type == 'demo':
        print('Using demo camera')
        from camera_handling.demo_video import start_demo_cam_thread
        start_demo_cam_thread(args.h, args.w, args.verbose['cam'])
    elif args.cam_type == 'genicam':
        print('Using generic Gen<i>Cam')
        from camera_handling.generic_genicam import start_generic_genicam_thread
        start_generic_genicam_thread(args.h, args.w, args.verbose['cam'])
    elif args.cam_type == 'load':
        image_path = args.image_load_path
        print(f'Loading images from: {image_path}')
        from camera_handling.load_images import image_load_loop
        _start_thread(image_load_loop, name="load-image-thread", args=(image_path,args.h, args.w)) 
    else:
        print('Using Allied Vision GeniCam (Vimba)')
        from camera_handling.allied_vision_genicam import vimba_loop
        vimba_thread = _start_thread(target=vimba_loop,name = "vimba-thread",args=( 0, 0, args.w,args.h,False, args.image_save_path))

    if args.dump_locally:
        from storage.save import dataframe_store_loop
        output_max_fps = 25
        storage_thread = _start_thread(target=dataframe_store_loop,
                                name = "storage-thread",
                                args=([output_max_fps, args.verbose['output'], 'output/']))

    signal.signal(signal.SIGINT, global_vars.cancel_signal)

    while global_vars.is_running:
        time.sleep(1)
    for t in global_vars.thread_holder:
        print(f'Joining {t}')
        t.join()
    global_vars.is_running_detection = False
    global_vars.handler_event.set()
    detection_thread.join()
    print("Stop.")
    exit(0)


if __name__ == '__main__':
    main()
