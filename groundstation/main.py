import tkinter as tk
from tkinter import ttk

from gui import StreamingFrame, AnnotationFrame, App
from ios.status_publisher import status_publisher_thread
from ios import *
from object_detector.workload_balancer import WorkloadBalancer, object_detector_factory

import threading
import config


def create_tk_window():
    print("# Create main window..")
    frame_annotation = None

    def annotation_enabled_callback(enabled: bool):
        if enabled:
            tk_notebook.tab(1, state="normal")
            frame_annotation.clear_classes()
        else:
            tk_notebook.tab(1, state="disabled")

    def tab_changed(event):
        index = tk_notebook.index(tk_notebook.select())
        if index == 0:
            frame_streaming.enable_redraw(True)
            frame_annotation.enable_redraw(False)
        elif index == 1:
            frame_streaming.enable_redraw(False)
            frame_annotation.enable_redraw(True)

    root = App(annotation_enabled_callback)

    print('# Create tk notebook..')
    tk_notebook = ttk.Notebook(root)
    tk_notebook.pack(fill=tk.BOTH, expand=True)

    frame_streaming = StreamingFrame(tk_notebook, root)
    root.frame_streaming = frame_streaming
    frame_annotation = AnnotationFrame(tk_notebook, root)
    frame_annotation._redraw = False
    root.frame_annotation = frame_annotation

    tk_notebook.add(frame_streaming, text="Stream")
    tk_notebook.add(frame_annotation, text="Annotate")
    tk_notebook.bind("<<NotebookTabChanged>>", tab_changed)
    frame_annotation.bind("<<NotebookTabChanged>>", lambda: print("annotation"))
    tk_notebook.tab(1, state="disabled")

    return root


def start_synchronizer(root, verbose):
    root.synchronizer = MetaDataAndVideoSynchronizer(root, verbose)

    thread = threading.Thread(target=root.synchronizer.exec, daemon=True, args=[])
    thread.start()

    return thread


def start_image_retrieve(root):
    use_udp=(config.VIDEO_TRANSMISSION == config.VIDEO_TRANSMISSION_ENUM.UDP)

    threads = []
    if config.COMBINE_TWO_VIDEO_STREAMS:
        root.stream_synchronizer = VideoStreamsSynchronizer(root, False)
        thread = threading.Thread(target=root.stream_synchronizer.exec, daemon=True, args=[])
        thread.start()
        threads.append(thread)

        submit_frame_funcs = [root.stream_synchronizer.add_left_image_for_frame, root.stream_synchronizer.add_right_image_for_frame]
        ports = config.ARM_DEVICE_UDP_STREAM_PORT
        for port, func in zip(ports, submit_frame_funcs):
            if use_udp:
                thread = threading.Thread(target=image_retrieve_thread_udp, daemon=True, args=[root, func, port])
                thread.start()
                threads.append(thread)
            else:
                raise Exception("Stream synchronization implement for UDP only")

    else:
        submit_frame_func = root.synchronizer.add_image_for_frame

        if not use_udp:
            thread = threading.Thread(target=image_retrieve_thread_rtsp, daemon=True, args=[root, submit_frame_func])
        else: 
            port = config.ARM_DEVICE_UDP_STREAM_PORT[0]
            thread = threading.Thread(target=image_retrieve_thread_udp, daemon=True, args=[root, submit_frame_func, port])
        thread.start()

    return threads


def start_roi_retrieve(root):
    thread = threading.Thread(target=metadata_retrieve_thread_tcp, daemon=True, args=[root])
    thread.start()

    return thread


def start_ground_station_status_publisher(root):
    thread = threading.Thread(target=status_publisher_thread, daemon=True, args=[root])
    thread.start()

    return thread


def start_detector_workload_balancer(root, visible_devices) -> WorkloadBalancer:
    workload_balancer = WorkloadBalancer(root, visible_devices, object_detector_factory)

    thread = threading.Thread(target=workload_balancer.exec, daemon=True, args=[])
    thread.start()

    return workload_balancer


if __name__ == '__main__':
    root = create_tk_window()
    root.running = True

    print("# Start synchronize thread..")
    thread_synchronizer = start_synchronizer(root, verbose=False)
    print("# Start image retrieve thread..")
    threads_img_retriever = start_image_retrieve(root)
    print("# Start roi retrieve thread..")
    thread_roi_retriever = start_roi_retrieve(root)
    print(f"# Start workload balancer with {len(config.VISIBLE_CUDA_DEVICES)} devices..")
    workload_balancer = start_detector_workload_balancer(root, config.VISIBLE_CUDA_DEVICES)

    print("# Start status packet publisher..")
    thread_status_publisher = start_ground_station_status_publisher(root)

    print("# Start main thread..")
    root.mainloop()

    root.running = False
    workload_balancer.stop()

    [t.join() for t in threads_img_retriever]
    thread_roi_retriever.join()
    thread_synchronizer.join()
