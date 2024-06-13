import PIL
from PIL import Image, ImageTk
from gui.app import App
import cv2

from .util import *
import config
import time

from .video_frame import VideoFrame


class StreamingFrame(tk.Frame):
    def __init__(self, parent, root: App):
        super(StreamingFrame, self).__init__(parent, bg='white')
        self.root = root
        self.rois = []
        self._redraw = True
        self.create_layout()

    def enable_redraw(self, enabled):
        self._redraw = enabled

        if enabled:
            self.video_frame._redraw()

    def create_layout(self):
        self.video_frame = PreviewFrame(self, self.root)
        self.video_frame.pack(side="top", fill="both", expand=True)
        tk.Label(self.video_frame, text="Test")

        self.menu_frame = StreamingMenuFrame(parent=self, root=self.root)
        self.menu_frame.pack(side="bottom", fill=tk.X)


class PreviewFrame(VideoFrame):
    def __init__(self, parent, root: App):
        super(PreviewFrame, self).__init__(parent)
        self.parent = parent
        self.root = root

        self.bind_mouse_event("<Button-1>", self._on_canvas_click)
        self.bind_mouse_event("<Button-3>", self._cancel_definition)
        self.bind_mouse_event("<Motion>", self._on_motion)
        self.bind_key("<Escape>", self._cancel_definition)

        # used to define custom roi areas
        self._custom_roi_definition = None

        self.fps = []

        OUT_VIDEO_FILE = '/tmp/detected_video.mp4'
        self.out = cv2.VideoWriter(OUT_VIDEO_FILE,
                                   cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
                                   30,
                                   (1022, 574))

    def _get_detections(self):
        detections = self.root.get_ready_detections()
        return detections

    def _redraw(self, verbose=False):
        """
            This method is responsible to redraw the main canvas
        :return:
        """

        start = time.time()
        image = self.root.get_ready_img()
        scaled_img = self._get_scaled_image(image)
        if scaled_img is not None:
            for roi in self.root.get_ready_rois():
                self.create_rectangle(scaled_img, roi, self._image_scale, self._scaled_del_btn_width,
                                      color=COLORS.BLACK)
            stop_roi = time.time()
            for roi in self.root.get_custom_rois_copy():
                self.create_rectangle(scaled_img, roi, self._image_scale, self._scaled_del_btn_width, color=COLORS.RED,
                                      delete_button=True)
            if self._custom_roi_definition is not None and len(self._custom_roi_definition) == 4:
                x1, y1, x2, y2 = self._custom_roi_definition
                cv2.rectangle(scaled_img, (x1, y1), (x2, y2), COLORS.GREEN)
            stop_croi = time.time()
            # add the detections
            detections = self._get_detections()
            for detection in detections:
                color = self.root.class_dictionary.get_color_for(detection.class_id)
                label = self.root.class_dictionary.get_label_for(detection.class_id)
                text = f"{label}: {round(detection.confidence, 1)}"
                self.create_rectangle(scaled_img, detection.to_numpy(), self._image_scale, self._scaled_del_btn_width,
                                      color=color,
                                      thickness=2, text=text)

            stop_d = time.time()
            self.tk_image = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(scaled_img))
            stop_c = time.time()
            self.redraw_image(self.tk_image)

        end = time.time()

        if verbose and scaled_img is not None:
            print(f"VISUALIZER - fps: {1 / (end - start)}")
            print(f"\tROI - fps: {1 / (stop_roi - start)}")
            print(f"\tCROI - fps: {1 / (stop_croi - stop_roi)}")
            print(f"\tDet - fps: {1 / (stop_d - stop_croi)}")
            print(f"\tConvert - fps: {1 / (stop_c - stop_d)}")
            print(f"\tVis - fps: {1 / (end - stop_c)}")

            self.fps.append(1 / (end - start))
            print(f"\t==> avg fps: {np.average(self.fps)}")

        if self.parent._redraw:
            self.after(max(int(1000 // 30 - 1000 * (end - start)), 10), lambda: self._redraw(verbose=verbose))

    def _check_for_delete_custom_roi(self, x, y):
        deleted = False
        # no list comprehension to be able to use deleted flag
        custom_rois = self.root.get_custom_rois_copy()
        self.root.clear_custom_rois()
        for roi in custom_rois:
            if self._is_in_delete_button(x, y, roi):
                deleted = True
            else:
                self.root.add_custom_roi(roi)
        return deleted

    def _on_canvas_click(self, event):
        if self._check_for_delete_custom_roi(event.x, event.y):
            # delete roi in _check_for_delete_custom_roi()
            pass
        elif self._custom_roi_definition is None:
            self._custom_roi_definition = np.array([event.x, event.y], dtype=int)
        else:
            custom_roi_definition = self.definition_from_rectangle(self._custom_roi_definition)
            self.root.add_custom_roi(custom_roi_definition)
            print(f"## New ROI definition: {custom_roi_definition}")
            self._custom_roi_definition = None

    def _cancel_definition(self, *args):
        self._custom_roi_definition = None

    def _on_motion(self, event):
        if self._custom_roi_definition is not None:
            if len(self._custom_roi_definition) == 4:
                self._custom_roi_definition[2] = event.x
                self._custom_roi_definition[3] = event.y
            else:
                self._custom_roi_definition = np.append(self._custom_roi_definition, [event.x, event.y])
                print(self._custom_roi_definition)


class StreamingMenuFrame(tk.Frame):
    def __init__(self, parent: StreamingFrame, root: App, **kwargs):
        super(StreamingMenuFrame, self).__init__(parent, bg='DarkGray', height=100, **kwargs)
        self.root = root
        self.parent = parent

        self.create_buttons()
        self.create_statusbar()
        self.create_certainty_slider()

    def freeze(self):
        if not self.root.paused_input:
            self.root.pause_input()
            self.root.annotation_enabled_callback(True)
            self.pause_btn.config(text='Play')
        else:
            self.root.continue_input()
            self.root.annotation_enabled_callback(False)
            self.pause_btn.config(text='Pause')

    def create_buttons(self):
        self.pause_btn = create_button(self, "Pause", self.freeze)
        self.pause_btn.place(anchor="e", relx=0.99, rely=0.20)

    def _update_status(self, in_loop=True):
        state = self.root.get_state()
        state_label = get_state_label(state)

        self.statusbar.config(state='normal')
        self.statusbar.delete(0, 'end')
        self.statusbar.insert(0, state_label)
        self.statusbar.config(state='readonly')

        if in_loop:
            self.after(10, self._update_status)

    def create_statusbar(self):
        self.statusbar = create_textbox(self, "...", editable=False)
        self.statusbar.place(anchor="w", relx=0.0, rely=0.85)
        self._update_status()

    def _update_certainty_threshold(self, x):
        print(f"Selected certainty {x}")
        config.CERTAINTY_THRESHOLD = float(x)

    def create_certainty_slider(self):
        current_value = tk.DoubleVar()
        self.certainty_slider = tk.Scale(self, from_=0.0, to=1.0, orient="horizontal", variable=current_value,
                                         resolution=.1,
                                         command=self._update_certainty_threshold)

        self.certainty_slider.set(config.CERTAINTY_THRESHOLD)
        self.certainty_slider.place(anchor="e", relx=0.99, rely=0.60)
        self.certainty_label = create_label(self, "Certainty: ")
        self.certainty_label.place(anchor="e", relx=.92, rely=.6)
