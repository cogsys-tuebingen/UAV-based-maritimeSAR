import torchvision
import torch
import time
import json
from collections import defaultdict
import os
from pycocotools.coco import COCO
import numpy as np
import cv2
import enum
import tqdm

import torchvision.transforms.functional as F


class ToTensor(object):
    def __call__(self, img, target):
        return F.to_tensor(img), target


class CutHandling(enum.Enum):
    REMOVE_SPLITS = "remove_splits",
    IGNORE_CUT_OBJECTS = "ignore_cut_objects"


class AddWholeImageHandling(enum.Enum):
    NOT = "not",
    AS_SPLIT = "as_split",
    FULL_RESOLUTION = "full_resolution"


def img2squares(w, h, sq, overlap=0.25):
    """
    create the tile positions for a image with the size w x h and the tile size
    of sq x sq
    
    @return a list of square splits (x1, y1, w, h)
    """

    if overlap == 0:
        xs = [i * sq for i in range(int(w / sq) + 1)]
        ys = [i * sq for i in range(int(h / sq) + 1)]

    else:
        # add the first and last box positions
        xs = [0, w - sq]
        ys = [0, h - sq]

        # add the overlapping boxes
        distance_between_x_splits = sq * (1 - overlap)
        distance_between_y_splits = sq * (1 - overlap)

        x_inter_split_num = int((xs[-1] - xs[0]) / distance_between_x_splits)
        if overlap == 0:
            x_inter_split_num -= 1
            x_inter_split_num = max(0, x_inter_split_num)

        y_inter_split_num = int((ys[-1] - ys[0]) / distance_between_y_splits)
        if overlap == 0:
            y_inter_split_num -= 1
            y_inter_split_num = max(0, y_inter_split_num)

        xs += [int((i * distance_between_x_splits + (sq / 2))) for i in range(x_inter_split_num)]
        ys += [int((i * distance_between_y_splits + (sq / 2))) for i in range(y_inter_split_num)]

        # move the last box position to the end
        xs.append(xs.pop(1))
        ys.append(ys.pop(1))

    xs, ys = np.array(xs), np.array(ys)
    p = np.stack(np.meshgrid(xs, ys)).T.astype(int)
    p_filled = np.zeros((p.shape[0], p.shape[1], p.shape[2] + 2), dtype=int)
    p_filled[:, :, :-2] = p
    p_filled[:, :, -2:] = sq
    p = p_filled

    return p


class FlightDetectionDataset(torchvision.datasets.VisionDataset):
    def __init__(self, root, annFile, max_whole_image_size, transform=None, target_transform=None,
                 remove_empty_imgs=True, split_size=512, use_dummy_images=False,
                 handle_cut_objects: CutHandling = False, cache_images_in_memory=False,
                 add_whole_image: AddWholeImageHandling= AddWholeImageHandling.NOT,
                 tile_overlapping=0.25):
        """
        @param max_whole_image_size: the max size of the whole image
        @param transform:
        @param target_transform:
        @param remove_empty_imgs: whether images with no object should be removed?
        @param split_size:
        @param use_dummy_images: don't load the image file
        @param handle_cut_objects: how to handle objects which are cut
        @param cache_images_in_memory: cache all imgs (doesn't speed up the process)
        @param add_whole_image: whether the whole image should also be a element of the dataset. There are three options:
                -   NOT: doesn't include the Whole image
                -   AS_SPLIT: The whole image is downsampled as split
                -   FULL_RESOLUTION: A second dataloader for full resoluton is needed (This dataloader will ignore the
                    whole images)
        """
        super(FlightDetectionDataset, self).__init__(root, transform=transform, target_transform=target_transform)
        self.remove_empty_imgs = remove_empty_imgs
        self.handle_cut_objects = handle_cut_objects

        self.split_size = split_size
        self.max_whole_image_size = max_whole_image_size
        self.tile_overlapping = tile_overlapping

        # don't load the image instead only return some empty black image
        self.use_dummy_images = use_dummy_images
        self.cache_images_in_memory = cache_images_in_memory

        self.add_whole_image = (add_whole_image == AddWholeImageHandling.AS_SPLIT)

        self._load_dataset(annFile)

    def _load_dataset(self, annotation_file):
        self.dataset, self.anns, self.cats, self.imgs = dict(), dict(), dict(), dict()
        self.shapes = []
        self.imgToAnns, self.catToImgs = defaultdict(list), defaultdict(list)
        self.idxToImg = []
        self.idxToPart = {}
        self.n_c = 0
        self.max_objs_in_image = 0

        if annotation_file is not None:
            print('# Loading annotations into memory and parse...')
            tic = time.time()
            dataset = json.load(open(annotation_file, 'r'))
            assert type(dataset) == dict, 'annotation file format {} not supported'.format(type(dataset))
            self.dataset = dataset
            self._createIndex()
            print('# Done (t={:0.2f}s)'.format(time.time() - tic))

    def _shrink_img_entry_to_max_size(self, img):
        if self.max_whole_image_size is None:
            img['whole_shrink_scale'] = 1.0
            return img

        width, height = img['width'], img['height']

        if height > width:
            scale = self.max_whole_image_size / height
            resized_height = self.max_whole_image_size
            resized_width = int(width * scale)
        else:
            scale = self.max_whole_image_size / width
            resized_height = int(height * scale)
            resized_width = self.max_whole_image_size

        if scale > 1.0:
            # we only shrink the larger images
            img['whole_shrink_scale'] = 1.0
            return img

        img['whole_shrink_scale'] = scale
        img['width'] = resized_width
        img['height'] = resized_height

        return img

    def _parse_categories(self):
        if isinstance(self.dataset['categories'], list):
            if isinstance(self.dataset['categories'][0], dict) and 'id' in self.dataset['categories'][0].keys():
                return {category['id']: category for category in self.dataset['categories']}
            else:
                return {_id: category for _id, category in enumerate(self.dataset['categories'])}

        if isinstance(self.dataset['categories'], dict):
            return self.dataset['categories']

    def _createIndex(self):
        # create index
        print('# Creating index...')
        imgs = {}
        shapes = {}
        self.img_parts = {}

        # FIXME there is a fixed number of 1000 splits per image!
        self.splits_per_img = 10000

        for img in self.dataset['images']:
            splitted_start_id = int(self.splits_per_img * img['id'])
            self.img_parts[splitted_start_id] = []

            img = self._shrink_img_entry_to_max_size(img)
            squares = img2squares(int(img['width']), int(img['height']), int(self.split_size),
                                  overlap=self.tile_overlapping)

            for line_id, line in enumerate(squares):
                for part_id, part in enumerate(line):
                    img_id = int(splitted_start_id + part_id + (line_id * len(line)))
                    imgs[img_id] = img.copy()
                    imgs[img_id]['width'] = part[2]
                    imgs[img_id]['height'] = part[3]
                    imgs[img_id]['part_info'] = {
                        'x_1': part[0],
                        'y_1': part[1],
                        'x_2': part[0] + part[2],
                        'y_2': part[1] + part[3],
                        'img_id_x': line_id,
                        'img_id_y': part_id,
                        'img_id': img_id,
                        'img_width': img['width'],
                        'img_height': img['height']
                    }
                    self.imgToAnns[img_id] = []
                    shapes[img_id] = ([imgs[img_id]['width'], imgs[img_id]['height']])
                    self.idxToPart[img_id] = (line_id, part_id)

                    self.img_parts[int(splitted_start_id)].append(
                        imgs[img_id]['part_info']
                    )

            if self.add_whole_image:
                # also add the whole img like in the power of tiling
                img_id = splitted_start_id + (self.splits_per_img - 1)
                imgs[img_id] = img.copy()
                imgs[img_id]['width'] = img['width']
                imgs[img_id]['height'] = img['height']
                imgs[img_id]['part_info'] = {
                    'x_1': 0,
                    'y_1': 0,
                    'x_2': img['width'],
                    'y_2': img['height'],
                    'img_id_x': -1,
                    'img_id_y': -1,
                    'img_id': img_id,
                    'img_width': img['width'],
                    'img_height': img['height']
                }
                self.imgToAnns[img_id] = []
                shapes[img_id] = ([imgs[img_id]['width'], imgs[img_id]['height']])
                self.idxToPart[img_id] = (-1, -1)

                self.img_parts[int(splitted_start_id)].append(
                    imgs[img_id]['part_info']
                )

        self.ann_id = 0

        for ann in tqdm.tqdm(self.dataset['annotations'], "Create annotations"):
            base_img_id = int(self.splits_per_img * ann['image_id'])
            whole_img_shrink_scale = imgs[base_img_id]['whole_shrink_scale']

            # search the correct img part
            bbox_x1 = ann['bbox'][0] * whole_img_shrink_scale
            bbox_y1 = ann['bbox'][1] * whole_img_shrink_scale
            bbox_width = ann['bbox'][2] * whole_img_shrink_scale
            bbox_height = ann['bbox'][3] * whole_img_shrink_scale

            bbox_x2 = bbox_x1 + bbox_width
            bbox_y2 = bbox_y1 + bbox_height

            fitting_parts = []
            for img_part in self.img_parts[base_img_id]:
                part_x1, part_y1, part_x2, part_y2 = img_part['x_1'], img_part['y_1'], img_part['x_2'], img_part['y_2']

                if bbox_x1 > part_x2 or bbox_x2 < part_x1:
                    continue

                if bbox_y1 > part_y2 or bbox_y2 < part_y1:
                    continue

                fitting_parts.append(img_part)

            # now add the annotation for each match
            for match in fitting_parts:
                part_x1, part_y1, part_x2, part_y2 = match['x_1'], match['y_1'], match['x_2'], match['y_2']

                _x_in_part = max(bbox_x1 - part_x1, 0)
                _y_in_part = max(bbox_y1 - part_y1, 0)

                _x2 = bbox_x1 + bbox_width
                _y2 = bbox_y1 + bbox_height

                _x2_in_part = min(_x2 - part_x1, part_x2 - part_x1)
                _y2_in_part = min(_y2 - part_y1, part_y2 - part_y1)

                actual_bbox_width = _x2_in_part - _x_in_part
                actual_bbox_height = _y2_in_part - _y_in_part

                is_cut = (int(actual_bbox_width) < int(bbox_width)) or (int(actual_bbox_height) < int(bbox_height))

                self._add_annotation(_x_in_part, _y_in_part, _x2_in_part - _x_in_part, _y2_in_part - _y_in_part,
                                     ann['category_id'],
                                     match['img_id'],
                                     is_cut=is_cut)

        self.empty_img_ids = [i for i in list(imgs.keys()) if len(self.imgToAnns[i]) == 0]
        print(f"Empty images[{len(self.empty_img_ids)}]: {self.empty_img_ids}")

        if self.remove_empty_imgs:
            # throw all images without annotation away
            for img_idx in self.empty_img_ids:
                    imgs.pop(img_idx)
                    self.imgToAnns.pop(img_idx)

        if self.handle_cut_objects == CutHandling.REMOVE_SPLITS:
            for img_idx in list(imgs.keys()):
                has_cut_annot = np.any([a['is_cut'] for a in self.imgToAnns[img_idx]])
                if has_cut_annot:
                    imgs.pop(img_idx)

        if self.handle_cut_objects == CutHandling.IGNORE_CUT_OBJECTS:
            for img_idx in list(imgs.keys()):
                has_uncut_annot = np.any([not a['is_cut'] for a in self.imgToAnns[img_idx]])
                if not has_uncut_annot:
                    imgs.pop(img_idx)

        self.n_c = len(self.dataset['categories'])

        self.cats = self._parse_categories()

        print('# Index created!')

        # create class members
        self.imgs = imgs
        self.shapes = np.array(shapes)
        self.max_objs_in_image = max([len(i) for i in self.imgToAnns.values()])
        self.idxToImg = list(self.imgs.keys())

        if self.cache_images_in_memory:
            self.cached_images = {}
            self._preload_images()

    def _preload_images(self):
        for img_meta in tqdm.tqdm(self.imgs.values(), desc="Preload images"):
            filename = img_meta['file_name']
            if filename not in self.cached_images.keys():
                img = cv2.imread(os.path.join(self.root, filename))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                self.cached_images[filename] = img

    def _add_annotation(self, x, y, width, height, label, img_id, is_cut):
        # ignore too small bboxes
        if width <= 2 or height <= 2:
            return

        self.anns[self.ann_id] = {
            'x': x,
            'y': y,
            'width': width,
            'height': height,
            'class_id': label,
            'is_cut': is_cut
        }
        self.imgToAnns[img_id].append(self.anns[self.ann_id])
        self.ann_id += 1

    def info(self):
        """
        Print information about the annotation file.
        :return:
        """
        for key, value in self.dataset['info'].items():
            print('{}: {}'.format(key, value))

    def load_annotations(self, image_index):
        # get ground truth annotations
        annotations = np.zeros((0, 6))

        for idx, a in enumerate(self.imgToAnns[image_index]):

            # some annotations have basically no width / height, skip them
            if a['width'] < 1 or a['height'] < 1:
                continue

            annotation = np.zeros((1, 6))
            annotation[0, :4] = [a['x'], a['y'], (a['x'] + a['width']), (a['y'] + a['height'])]
            annotation[0, 4] = a['class_id']
            annotation[0, 5] = a['is_cut']
            annotations = np.append(annotations, annotation, axis=0)

        return annotations

    def _get_whole_img_for_split_img_id(self, img_id):
        """
            This used the real img id, because there could be no mapping entry
        """
        img, img_meta = self._load_image(img_id)
        shrink_scale = img_meta['whole_shrink_scale']

        if shrink_scale != 1.0:
            img = cv2.resize(img, (img_meta['part_info']['img_width'], img_meta['part_info']['img_height']))

        return img

    def _load_image(self, img_idx):
        img_meta = self.imgs[img_idx]
        path = img_meta['file_name']

        if self.cache_images_in_memory:
            if path in self.cached_images.keys():
                img = self.cached_images[path].copy()
            else:
                img = cv2.imread(os.path.join(self.root, path))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                self.cached_images[path] = img.copy()
        else:
            img = cv2.imread(os.path.join(self.root, path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if img is None:
            raise Exception("Could not load img: %s" % path)

        return img, img_meta

    def _get_img(self, index):
        img_idx = self.idxToImg[index]

        target = self.load_annotations(img_idx)

        if self.use_dummy_images:
            img = np.zeros((self.imgs[img_idx]['width'], self.imgs[img_idx]['height'], 3))
        else:
            img, img_meta = self._load_image(img_idx)

            shrink_scale = img_meta['whole_shrink_scale']
            if shrink_scale != 1.0:
                img = cv2.resize(img, (img_meta['part_info']['img_width'], img_meta['part_info']['img_height']))

            # get correct part of the image
            _part_x, _part_y = self.idxToPart[img_idx]
            img = img[img_meta['part_info']['y_1']:img_meta['part_info']['y_2'],
                  img_meta['part_info']['x_1']:img_meta['part_info']['x_2'], :]

            if self.handle_cut_objects == CutHandling.IGNORE_CUT_OBJECTS:
                # set the cut objects to black and remove their labels
                is_cut_mask = target[:, 5] > 0
                for (_x, _y, _x2, _y2) in target[is_cut_mask, :4]:
                    img[int(_y): int(_y2), int(_x):int(_x2), :] = 0
                target = target[np.logical_not(is_cut_mask)]

            if img.shape[0] < self.split_size or img.shape[1] < self.split_size:
                # if the original image is smaller then the split size, there can be problems
                # it is filled up with black
                # for example for the vis drone dataset
                extended_img = np.zeros((self.split_size, self.split_size, 3), dtype=np.uint8)
                extended_img[:img.shape[0], :img.shape[1]] = img
                img = extended_img

        # remove the is_cut column
        target = target[:, :5]

        sample = {'img': torch.from_numpy(img), 'annot': torch.from_numpy(target), 'img_id': img_idx,
                  'meta': {
                      'max_objs_in_image': self.max_objs_in_image
                  }}
        if self.transform:
            sample = self.transform(sample)

        return sample

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: Tuple (image, target). target is the object returned by ``coco.loadAnns``.
        """
        return self._get_img(index)

    def __len__(self):
        return len(self.idxToImg)

    def num_classes(self):
        return self.n_c


def make_flightdb_transforms(image_set, image_size):
    pass


def convert_to_coco_api(ds):
    coco_ds = COCO()
    # annotation IDs need to start at 1, not 0, see torchvision issue #1530
    ann_id = 1
    dataset = {'images': [], 'categories': [], 'annotations': []}
    categories = set()
    for img_idx in range(len(ds)):
        # find better way to get target
        # targets = ds.get_annotations(img_idx)
        batch = ds.get_in_org_size(img_idx)
        img, targets = batch['img'].float(), batch['annot']
        image_id = batch['img_id']
        img_dict = {}
        img_dict['id'] = image_id
        img_dict['height'] = img.shape[-2]
        img_dict['width'] = img.shape[-1]
        dataset['images'].append(img_dict)
        bboxes = targets[:, :4]
        bboxes[:, 2:] -= bboxes[:, :2]
        bboxes = bboxes
        labels = targets[:, 4].tolist()
        areas = (bboxes[:, 2] * bboxes[:, 3]).tolist()
        num_objs = len(bboxes)
        iscrowd = [False for _ in range(num_objs)]
        for i in range(num_objs):
            ann = {}
            ann['image_id'] = image_id
            ann['bbox'] = bboxes[i].tolist()
            ann['category_id'] = int(labels[i])
            categories.add(int(labels[i]))
            ann['area'] = areas[i]
            ann['iscrowd'] = iscrowd[i]
            ann['id'] = ann_id
            dataset['annotations'].append(ann)
            ann_id += 1
    dataset['categories'] = [{'id': int(i)} for i in sorted(categories)]
    coco_ds.dataset = dataset
    coco_ds.createIndex()
    return coco_ds


if __name__ == '__main__':
    data_path = '/data/datasets/vis_drone'
    os.makedirs("/tmp/test", exist_ok=True)

    PATHS = {
        "train": (os.path.join(data_path, "images", "train"),
                  os.path.join(data_path, "annotations", 'instances_train.json')),
        "val": (os.path.join(data_path, "images", "val"),
                os.path.join(data_path, "annotations", 'instances_val.json')),
    }

    img_folder = PATHS['train'][0]
    ann_file = PATHS['train'][1]

    dataset = FlightDetectionDataset(img_folder, ann_file,
                                     remove_empty_imgs=True, split_size=512,
                                     max_whole_image_size=1024,
                                     tile_overlapping=0)

    import matplotlib.pyplot as plt
    import data_loader.bounding_box_util as bb
    import tqdm

    np.random.seed(0)

    for i in tqdm.tqdm(range(20)):
        img_id = list(dataset.img_parts.keys())[np.random.randint(0, len(dataset.img_parts.keys()))]
        splits = dataset.img_parts[img_id]

        not_empty_splits = [s for s in splits if s['img_id'] in dataset.idxToImg]

        whole_img = dataset._get_whole_img_for_split_img_id(not_empty_splits[0]['img_id'])

        split_id = 0

        for split in splits:
            is_used = split['img_id'] in dataset.idxToImg

            whole_img = bb.add_bbox_xyxy(whole_img, split['x_1'], split['y_1'], split['x_2'], split['y_2'],
                                         color='blue' if is_used else "gray")
            #whole_img = bb.add_filled_bbox_xyxy(whole_img, split['x_1'], split['y_1'], split['x_2'], split['y_2'],
#                                                color='gray')

            if split['img_id'] in dataset.idxToImg:
                sample = dataset[dataset.idxToImg.index(split['img_id'])]
                split_img = sample['img'].numpy()
                for an in sample['annot']:
                    # if an[4] == 5:
                    #     continue
                    whole_img = bb.add_bbox_xyxy(whole_img, split['x_1'] + an[0],
                                                 split['y_1'] + an[1],
                                                 split['x_1'] + an[2],
                                                 split['y_1'] + an[3])
                    split_img = bb.add_bbox_xyxy(split_img, an[0],
                                                 an[1],
                                                 an[2],
                                                 an[3], color='orange')
                plt.imsave('/tmp/test/%i_split_%i.png' % (i, split_id), split_img)
            split_id += 1


        plt.imsave('/tmp/test/%i_whole.png' % i, whole_img)
        # bb.show_img(whole_img)
