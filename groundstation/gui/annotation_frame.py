import copy
import tkinter as tk
import PIL
import cv2
import numpy as np

from . import util
from .util import create_button, create_class_listbox
from .app import App
from .video_frame import VideoFrame


class ClassDefinition:
    def __init__(self, class_name: str, definition: np.ndarray):
        self.class_name = class_name
        self.definition = definition


class AnnotationFrame(tk.Frame):
    def __init__(self, parent, root: App):
        super(AnnotationFrame, self).__init__(parent, bg='white')

        self.menu_frame = None
        self.main_frame = None
        self.root = root
        self.current_class = root.class_dictionary.get_labels()[0]
        self._redraw = True

        self.create_layout()

    def enable_redraw(self, enabled):
        self._redraw = enabled
        self.main_frame._redraw()

    def create_layout(self):
        self.main_frame = PreviewFrame(self, self.root)
        self.main_frame.pack(side="top", fill="both", expand=True)
        tk.Label(self.main_frame, text="Test")

        self.menu_frame = AnnotationMenuFrame(parent=self, root=self.root,
                                              classes=self.root.class_dictionary.get_labels())
        self.menu_frame.pack(side="bottom", fill=tk.X)

    def class_changed(self, class_index):
        self.current_class = self.root.class_dictionary.get_labels()[class_index]

    def save(self):
        self.main_frame.save()

    def clear_classes(self):
        self.main_frame.clear_classes()


class PreviewFrame(VideoFrame):
    def __init__(self, parent, root: App):
        super(PreviewFrame, self).__init__(parent)
        self.class_definitions = []
        self.parent = parent
        self.root = root

        self.bind_mouse_event("<Button-1>", self._on_canvas_click)
        self.bind_mouse_event("<Button-3>", self.cancel_definition)
        self.bind_mouse_event("<Motion>", self._on_motion)
        self.bind_key("<Escape>", self.cancel_definition)

        self._current_class_def = None

        self._redraw()

    def _redraw(self):
        """
            This method is responsible to redraw the main canvas
        :return:
        """
        image = self.root.get_ready_img()
        scaled_img = self._get_scaled_image(image)

        if scaled_img is not None:
            if self._current_class_def is not None and len(self._current_class_def.definition) == 4:
                x1, y1, x2, y2 = self._current_class_def.definition
                class_id = self.root.class_dictionary.get_id_for(self.parent.current_class)
                color = self.root.class_dictionary.get_color_for(class_id)
                cv2.rectangle(scaled_img, (x1, y1), (x2, y2), color)
                cv2.putText(scaled_img, self._current_class_def.class_name, (x1, y1 - 5), cv2.FONT_HERSHEY_PLAIN, 0.9,
                            color)
            for class_definition in self.class_definitions:
                class_id = self.root.class_dictionary.get_id_for(class_definition.class_name)
                color = self.root.class_dictionary.get_color_for(class_id)
                self.create_rectangle(scaled_img, class_definition.definition, self._image_scale,
                                      self._scaled_del_btn_width,
                                      color, True, 1, class_definition.class_name)

            self.tk_image = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(scaled_img))
            self.redraw_image(self.tk_image)

        if self.parent._redraw:
            self.after(10, self._redraw)

    def _check_for_delete_custom_class(self, x, y):
        deleted = False
        # no list comprehension to be able to use deleted flag
        class_definitions = copy.deepcopy(self.class_definitions)
        self.class_definitions.clear()
        for class_definition in class_definitions:
            if self._is_in_delete_button(x, y, class_definition.definition):
                deleted = True
            else:
                self.class_definitions.append(class_definition)
        return deleted

    def _on_canvas_click(self, event):
        if self._check_for_delete_custom_class(event.x, event.y):
            # delete roi in _check_for_delete_custom_roi()
            pass
        elif self._current_class_def is None:
            definition = np.array([event.x, event.y], dtype=int)
            self._current_class_def = ClassDefinition(self.parent.current_class, definition)
        else:
            self._current_class_def.definition = self.definition_from_rectangle(self._current_class_def.definition)
            print(f"## New class definition: {self._current_class_def}")
            self.class_definitions.append(self._current_class_def)
            self._current_class_def = None

    def cancel_definition(self, *args):
        self._current_class_def = None

    def _on_motion(self, event):
        if self._current_class_def is not None:
            if len(self._current_class_def.definition) == 4:
                self._current_class_def.definition[2] = event.x
                self._current_class_def.definition[3] = event.y
            else:
                self._current_class_def.definition = np.append(self._current_class_def.definition, [event.x, event.y])

    def _on_canvas_key(self, event):
        if event.keycode == 9:
            self._current_class_def = None

    def save(self):
        util.save_annotations(self.root.get_img(), self.class_definitions, self.root.class_dictionary)

    def clear_classes(self):
        self.class_definitions = []


class AnnotationMenuFrame(tk.Frame):
    def __init__(self, parent: AnnotationFrame, root: App, classes=None, **kwargs):
        super(AnnotationMenuFrame, self).__init__(parent, bg='DarkGray', height=100, **kwargs)
        if classes is None:
            classes = []
        self.root = root
        self.parent = parent

        self.create_buttons()
        self.create_class_listmenu(classes)

    def create_buttons(self):
        save_btn = create_button(self, "Save", self.parent.save)
        save_btn.place(anchor="e", relx=0.99, rely=0.80)

    def create_class_listmenu(self, classes):
        classes_menu = create_class_listbox(self, classes, callback=self.parent.class_changed)
        # classes_menu.place(anchor="c", relx=.5, rely=.2)
        classes_menu.place(anchor="n", relx=.5, height=100)
