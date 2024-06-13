import argparse
import json
import os
import cv2
import tqdm
import matplotlib.pyplot as plt
import numpy as np

import data_loader.bounding_box_util as bbox_util
from data_loader.flight_detection_dataset import FlightDetectionDataset



class Holder:
    def __init__(self, root_path, annotation_file):
        self.root_path = root_path
        self.annotation_file = annotation_file

        self.load_dataset(self.annotation_file)

    def load_dataset(self, annotation_file):
        if annotation_file is None:
            return

        self.dataset = json.load(open(annotation_file, 'r'))
        assert type(self.dataset) == dict, 'annotation file format {} not supported'.format(type(self.dataset))

        self.anns, self.imgs = dict(), dict()
        self.imgToAnns = {}

        for img in self.dataset['images']:
            img_id = img['id']
            self.imgs[img_id] = img
            self.imgToAnns[img_id] = []

        for ann in self.dataset['annotations']:
            self.anns[ann['id']] = ann
            self.imgToAnns[ann['image_id']].append(self.anns[ann['id']])

    def resize(self, max_size, output_path):
        for img in tqdm.tqdm(list(self.imgs.values())):
            anns = self.imgToAnns[img['id']]

            image_data = cv2.imread(os.path.join(self.root_path, img['file_name']))
            height, width, _ = image_data.shape

            if height > width:
                scale = max_size / height
                resized_height = max_size
                resized_width = int(width * scale)
            else:
                scale = max_size / width
                resized_height = int(height * scale)
                resized_width = max_size

            image_data = cv2.resize(image_data, (resized_width, resized_height))

            cv2.imwrite(os.path.join(output_path, img['file_name']), image_data)

            img['height'] = resized_height
            img['width'] = resized_width

            for ann in anns:
                ann['bbox'] = [int(v * scale) for v in ann['bbox']]

    def save_dataset(self, annotation_file):
        json.dump(self.dataset, open(annotation_file, 'w'))


def get_args():
    parser = argparse.ArgumentParser("Downsample a data set")
    parser.add_argument("--max_size", type=int, required=True)
    parser.add_argument("--input_path", type=str, required=True, help="the root folder of dataset")
    parser.add_argument("--output_path", type=str, required=True, help="the root folder of dataset")

    return parser.parse_args()


def create_if_not_exists(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


if __name__ == '__main__':
    opt = get_args()

    PATHS = {
        "train": (os.path.join(opt.input_path, "images", "train"),
                  os.path.join(opt.input_path, "annotations", 'instances_train.json')),
        "val": (os.path.join(opt.input_path, "images", "val"),
                os.path.join(opt.input_path, "annotations", 'instances_val.json')),
    }

    OUTPUT_PATHS = {
        "train": (os.path.join(opt.output_path, "images", "train"),
                  os.path.join(opt.output_path, "annotations", 'instances_train.json')),
        "val": (os.path.join(opt.output_path, "images", "val"),
                os.path.join(opt.output_path, "annotations", 'instances_val.json')),
    }

    create_if_not_exists(opt.output_path)

    for mode in ['train', 'val']:
        img_folder, ann_file = PATHS[mode]
        holder = Holder(img_folder, ann_file)
        output_img_folder, output_ann_file = OUTPUT_PATHS[mode]
        create_if_not_exists(output_img_folder)
        create_if_not_exists(os.path.join(opt.output_path, "annotations"))
        holder.resize(opt.max_size, output_img_folder)
        holder.save_dataset(output_ann_file)


    # visualize an example
    img_folder, ann_file = OUTPUT_PATHS['train']
    dataset = FlightDetectionDataset(img_folder, ann_file, opt.max_size)
    for id in np.random.randint(0, len(dataset), 15):
        sample = dataset[id]
        img, ann = sample['img'], sample['annot']

        for bbox in ann:
            img = bbox_util.add_bbox_xyxy(img, bbox[0], bbox[1], bbox[2], bbox[3])
        plt.figure()
        plt.imshow(img)
        plt.show()






