import tkinter as tk
import cv2
import numpy as np

from gui.util import COLORS


class VideoFrame(tk.Frame):
    def __init__(self, parent):
        super(VideoFrame, self).__init__(parent, bg='white')

        self._image_padding_x = 0
        self._image_padding_y = 0
        self._image_layer = tk.Canvas(self)
        self._image_layer_image_id = self._image_layer.create_image(0, 0, anchor=tk.NW)

        self._image_layer.pack(fill=tk.BOTH, expand=True, side=tk.TOP)
        self._image_scale = 1

        self._tk_image = None

        self._image_layer_target_size = None
        self.del_btn_width = 20
        self._scaled_del_btn_width = self.del_btn_width

    def create_rectangle(self, image, coordinates, image_scale=1, scaled_del_btn_width=20, color=COLORS.BLACK,
                         delete_button=False, thickness=1, text=None):
        img_height = image.shape[0]
        img_width = image.shape[1]
        xmin, ymin, width, height = (image_scale * coordinates).astype(int)
        xmax = clip_to_value(xmin + width, img_width)
        ymax = clip_to_value(ymin + height, img_height)
        xmin = clip_to_value(xmin, img_width)
        ymin = clip_to_value(ymin, img_height)
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, thickness)
        if delete_button:
            cv2.rectangle(image, (xmax - scaled_del_btn_width, ymin + scaled_del_btn_width),
                          (xmax, ymin),
                          COLORS.RED, cv2.FILLED)
            cv2.line(image, (xmax - scaled_del_btn_width, ymin + scaled_del_btn_width), (xmax, ymin),
                     COLORS.WHITE,
                     2)
            cv2.line(image, (xmax, ymin + scaled_del_btn_width), (xmax - scaled_del_btn_width, ymin),
                     COLORS.WHITE,
                     2)
        if text:
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_PLAIN, 1, 0)
            cv2.putText(image, text, (xmin + 2, (ymin - 2 * text_size[1]) + 2),
                        cv2.FONT_HERSHEY_PLAIN, 1,
                        color, 1)

    def _get_scaled_image(self, image):
        recalculate_scale = self._image_layer_target_size != (
            self._image_layer.winfo_width(), self._image_layer.winfo_height())
        if image is not None:
            frame_height = self._image_layer.winfo_height()
            frame_width = self._image_layer.winfo_width()
            if recalculate_scale and frame_width != 1 and frame_height != 1:  # avoid error if not image_layer.pack() called
                self._image_layer.update()
                self._image_layer_target_size = (self._image_layer.winfo_width(), self._image_layer.winfo_height())
                # scale image to fit into the canvas
                self._image_scale = min(frame_height / image.shape[0], frame_width / image.shape[1])
                self._scaled_del_btn_width = int(self.del_btn_width * (self._image_scale / 1.5))
                # center canvas
                self._image_padding_x = int((frame_width - (self._image_scale * image.shape[1])) / 2)
                self._image_padding_y = int((frame_height - (self._image_scale * image.shape[0])) / 2)

            return cv2.resize(image, (int(self._image_scale * image.shape[1]), int(
                self._image_scale * image.shape[0])))
        else:
            return None

    def redraw_image(self, image):
        self._image_layer.delete(self._image_layer_image_id)
        self._image_layer_image_id = self._image_layer.create_image(self._image_padding_x,
                                                                    self._image_padding_y,
                                                                    image=image, anchor=tk.NW)

    def definition_from_rectangle(self, definition):
        xmin = min(definition[0], definition[2])
        xmax = max(definition[0], definition[2])
        ymin = min(definition[1], definition[3])
        ymax = max(definition[1], definition[3])
        # to xmin, ymin, width, height
        res_definition = np.array([xmin, ymin, xmax - xmin, ymax - ymin])
        res_definition = res_definition // self._image_scale
        res_definition = res_definition.astype(int)
        return res_definition

    def _is_in_delete_button(self, x, y, roi):
        y1 = roi[1] * self._image_scale
        x2 = (roi[0] + roi[2]) * self._image_scale
        return x2 - self._scaled_del_btn_width < x < x2 and \
               y1 + self._scaled_del_btn_width > y > y1

    def bind_key(self, sequence, callback):
        self._image_layer.bind(sequence, callback)

    def bind_mouse_event(self, sequence, callback):
        """
        Bind button events and apply padding to them
        @param sequence:
        @param callback:
        """
        self._image_layer.bind(sequence, lambda e: callback(self._padding_to_mouse(e)))

    def _padding_to_mouse(self, event) -> (int, int):
        """
        Applies padding to mouse position
        @param event:
        @return:
        """
        event.x = event.x - self._image_padding_x
        event.y = event.y - self._image_padding_y
        return event


def clip_to_value(v, value):
    return min(max(0, v), value)
