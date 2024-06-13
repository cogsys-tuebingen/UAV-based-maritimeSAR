import threading
import queue
import time

from typing import Callable, Any

from object_detector import Detection, TrainedModelInterface
from object_detector.models import FasterRCNN
from gui.app import App
import config


class WorkloadBalancer:
    def __init__(self, root: App, visible_devices, gpu_thread_factory: Callable[[Any, str], None]):
        self.root = root
        self.visible_devices = visible_devices
        self.input_queue = queue.Queue(len(visible_devices) * 2)
        self.output_queue = queue.Queue(len(visible_devices) * 2)
        self.threads = []
        self.running = True

        self._create_threads(gpu_thread_factory)
        self.last_seen_frame_id = 0
        self.frame_id = 0

    def _create_threads(self, factory):
        for device in self.visible_devices:
            self.threads.append(factory(self, device))

    def exec(self):
        self.running = True
        [t.start() for t in self.threads]

        while self.running:
            start = time.time()

            cur_frame_id = self.frame_id
            self.frame_id += 1

            if self.input_queue.full():
                self.input_queue.get_nowait()

            # add the next sample to the input queue
            if config.SYNC_PREDICTIONS:
                sample = (*self.root.get_roi_region_imgs(), self.root.get_rois(), self.root.get_img(), cur_frame_id)
            else:
                sample = (*self.root.get_roi_region_imgs(), None, None, cur_frame_id)

            self.input_queue.put(sample)

            try:
                out = self.output_queue.get_nowait()
                detections, rois, img, frame_id = out

                if self.last_seen_frame_id < frame_id:
                    if config.SYNC_PREDICTIONS:
                        self.root.set_processed_data(img=img, rois=rois, detections=detections)
                    else:
                        self.root.set_processed_data(detections=detections)

                    self.last_seen_frame_id = frame_id

            except queue.Empty:
                pass

            stop = time.time()
            duration = stop - start

            time.sleep(max(0, (1 / 60 - duration)))

    def stop(self):
        self.running = False
        [t.join() for t in self.threads]

    def get_detection_confidence_threshold(self):
        return self.root.get_detection_confidence_threshold()

    def submit_predictions(self, detections, rois, img, frame_id):
        try:
            if self.output_queue.full():
                self.output_queue.get_nowait()

            self.output_queue.put((detections, rois, img, frame_id))
        except queue.Full:
            pass

    def get_input(self):
        try:
            return self.input_queue.get_nowait()
        except queue.Empty:
            return None


def load_object_detector(device) -> TrainedModelInterface:
    print(f"# [{device}] Load object detector..")
    model = None

    try:
        model = FasterRCNN(config.OBJECT_DETECTOR_PTH,
                           num_classes=config.NUM_CLASSES,
                           device_id=device,
                           backbone=config.OBJECT_DETECTOR_BACKBONE)

    except Exception as e:
        print(e)

    if model is None:
        print(f"# [{device}] Could not load object detector checkpoint: {config.OBJECT_DETECTOR_PTH}")

    return model


def object_detector_factory(workbalancer: WorkloadBalancer, device: str):
    model = load_object_detector(device)

    def detector_thread(workbalancer: WorkloadBalancer):
        while workbalancer.running:
            start = time.time()

            pred_input = workbalancer.get_input()

            if pred_input is not None:

                roi_imgs, roi_offsets, rois, img, cur_frame_id = pred_input

                final_detections = []
                if len(roi_imgs) > 0:
                    # FIXME : remove
                    detections_for_rois = model.inference(roi_imgs[:3], workbalancer.get_detection_confidence_threshold())
                    # create the finale detection by using the roi_offsets
                    for offsets, detections in zip(roi_offsets, detections_for_rois):
                        for detection in detections:
                            detection.add_offset(offsets[0], offsets[1])
                            detection.clip_to(offsets[0], offsets[1], offsets[2], offsets[3])

                            if detection.is_empty():
                                continue
                            final_detections.append(detection)

                workbalancer.submit_predictions(detections=final_detections, rois=rois, img=img, frame_id=cur_frame_id)

            stop = time.time()
            duration = stop - start

            time.sleep(max(0, (1 / 60 - duration)))

        print(f"# [{device}] Stopped worker")

    if model is None:
        raise ChildProcessError(f"# [{device}] Could not load object detector")

    thread = threading.Thread(target=detector_thread, daemon=True, args=[workbalancer])
    return thread
