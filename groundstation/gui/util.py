import json
import os
import tkinter as tk
import enum
from datetime import datetime
from pathlib import Path
from tkinter import filedialog
from tkinter.messagebox import askyesno

import cv2
import numpy as np

import config


class AppState(enum.Enum):
    CONNECTING = 1
    CONNECTED = 2
    ERROR = 0


class COLORS:
    RED = (255, 0, 0)
    WHITE = (255, 255, 255)
    GREEN = (0, 255, 0)
    BLACK = (0, 0, 0)



def get_state_label(state):
    if state.value == AppState.CONNECTING.value:
        return "Connecting.."
    elif state.value == AppState.CONNECTED.value:
        return "Connected"
    elif state.value == AppState.ERROR.value:
        return "Error!"

    return "Unknown"


def create_button(parent, name, callback=None, textfont=("Arial", 10)):
    button = tk.Button(parent, text=name, command=callback)
    button.config(font=textfont)

    return button


def create_label(parent, name, textfont=("Arial", 12), color='black'):
    label = tk.Label(parent, text=name, fg=color)
    label.config(font=textfont)

    return label


def create_textbox(parent, name, textfont=("Arial", 12), color='black', editable=True):
    text = tk.Entry(parent, fg=color)
    text.insert('end', name)
    text.config(state='normal' if editable else 'readonly')
    text.config(font=textfont)

    return text


def create_check_box(parent, name, value, textfont=("Arial", 10)):
    box = tk.Checkbutton(parent, text=name, variable=value)
    box.config(font=textfont)

    return box


def create_listbox(parent, height, width, content="None", selectionmode=tk.SINGLE, textfont=("Arial", 8)):
    listbox = tk.Listbox(parent, bg='gray', fg='white', height=height,
                         selectbackground='purple', selectmode=selectionmode, width=width)
    listbox.delete(0, tk.END)
    listbox.config(font=textfont)
    listbox.insert(tk.END, *content)

    return listbox


def create_class_listbox(parent, items, width=20, font=("Arial", 10), callback=None):
    variable = tk.StringVar(value=items)
    listbox = tk.Listbox(parent, listvariable=variable)
    listbox.config(width=width, font=font)
    listbox.select_set(0)
    listbox.bind('<<ListboxSelect>>', lambda event: callback(listbox.curselection()[0]))
    return listbox


def save_annotations(image, class_definitions, class_dictionary):
    dir_name = filedialog.askdirectory()
    if not dir_name:
        return
    image_id = 1
    images_path = Path(os.path.join(dir_name, "images/train"))
    instances_path = Path(os.path.join(dir_name, "annotations/instances_train.json"))
    if not instances_path.exists():
        create_new = askyesno(title='New dataset',
                              message='No existing dataset found. Are you sure about creating a new one?')
        if not create_new:
            return
    if images_path.exists():
        image_names = os.listdir(images_path)
        max_image_id = len(image_names)
        for image_file_name in image_names:  # avoid same name if error in dataset
            name = image_file_name.split(".")[0]
            if int(name) >= max_image_id:
                max_image_id = int(name)
        image_id = max_image_id + 1
    else:
        images_path.mkdir(parents=True)
    image_file_name = f"{image_id}.png"
    print(f"Saving image in {images_path.joinpath(image_file_name)}")
    cv2.imwrite(str(images_path.joinpath(image_file_name)), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    image_description = {'id': image_id, 'file_name': image_file_name, 'source': {"new": True},
                         'date_time': datetime.now().isoformat(), 'width:': image.shape[1],
                         "height": image.shape[0]}
    max_annotation_id = 0
    instances = dict()
    if instances_path.exists():
        with open(instances_path, 'r') as f:
            instances = json.load(f)
    else:
        os.mkdir(os.path.join(dir_name, "annotations"))
        instances["categories"] = config.CLASSES
        instances["images"] = []
        instances["annotations"] = []
    for class_definition in class_definitions:
        annotation = {'id': max_annotation_id + 1, 'bbox': class_definition.definition,
                      'area': class_definition.definition[2] * class_definition.definition[3],
                      'category_id': class_dictionary.get_id_for(class_definition.class_name),
                      'image_id': image_id}
        instances["annotations"].append(annotation)
        max_annotation_id += 1
    instances["images"].append(image_description)
    with open(instances_path, 'w') as f:
        json.dump(instances, f, cls=NpEncoder)
    print(f"saved instances in {str(instances_path)}")


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


def create_class_listbox(parent, items, width=20, font=("Arial", 10), callback=None):
    variable = tk.StringVar(value=items)
    listbox = tk.Listbox(parent, listvariable=variable)
    listbox.config(width=width, font=font)
    listbox.select_set(0)
    listbox.bind('<<ListboxSelect>>', lambda event: callback(listbox.curselection()[0]))
    return listbox


def save_annotations(image, class_definitions, class_dictionary):
    dir_name = filedialog.askdirectory()
    if not dir_name:
        return
    image_id = 1
    images_path = Path(os.path.join(dir_name, "images/train"))
    instances_path = Path(os.path.join(dir_name, "annotations/instances_train.json"))
    if not instances_path.exists():
        create_new = askyesno(title='New dataset',
                              message='No existing dataset found. Are you sure about creating a new one?')
        if not create_new:
            return
    if images_path.exists():
        image_names = os.listdir(images_path)
        max_image_id = len(image_names)
        for image_file_name in image_names:  # avoid same name if error in dataset
            name = image_file_name.split(".")[0]
            if int(name) >= max_image_id:
                max_image_id = int(name)
        image_id = max_image_id + 1
    else:
        images_path.mkdir(parents=True)
    image_file_name = f"{image_id}.png"
    print(f"Saving image in {images_path.joinpath(image_file_name)}")
    cv2.imwrite(str(images_path.joinpath(image_file_name)), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    image_description = {'id': image_id, 'file_name': image_file_name, 'source': {"new": True},
                         'date_time': datetime.now().isoformat(), 'width:': image.shape[1],
                         "height": image.shape[0]}
    max_annotation_id = 0
    instances = dict()
    if instances_path.exists():
        with open(instances_path, 'r') as f:
            instances = json.load(f)
    else:
        os.mkdir(os.path.join(dir_name, "annotations"))
        instances["categories"] = config.CLASSES
        instances["images"] = []
        instances["annotations"] = []
    for class_definition in class_definitions:
        annotation = {'id': max_annotation_id + 1, 'bbox': class_definition.definition,
                      'area': class_definition.definition[2] * class_definition.definition[3],
                      'category_id': class_dictionary.get_id_for(class_definition.class_name),
                      'image_id': image_id}
        instances["annotations"].append(annotation)
        max_annotation_id += 1
    instances["images"].append(image_description)
    with open(instances_path, 'w') as f:
        json.dump(instances, f, cls=NpEncoder)
    print(f"saved instances in {str(instances_path)}")


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)
