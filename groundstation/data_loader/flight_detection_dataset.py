import torchvision
import time
import json
from collections import defaultdict
import os
from pycocotools.coco import COCO
import numpy as np
import cv2
import torch
import pandas
import tqdm

import torchvision.transforms.functional as F


class ToTensor(object):
    def __call__(self, img, target):
        return F.to_tensor(img), target


class FlightDetectionDataset(torchvision.datasets.VisionDataset):
    def __init__(self, root, annFile, metaFile, max_whole_image_size=None, transform=None, target_transform=None, transforms=None, use_dummy_images=False, cache_images_in_memory=False):
        super(FlightDetectionDataset, self).__init__(root, transforms, transform, target_transform)

        self.max_whole_image_size = max_whole_image_size

        # don't load the image instead only return some empty black image
        self.use_dummy_images = use_dummy_images
        self.cache_images_in_memory = cache_images_in_memory

        self._load_dataset(annFile, metaFile)

    def _load_dataset(self, annotation_file, meta_file):
        self.dataset, self.anns, self.cats, self.imgs = dict(), dict(), dict(), dict()
        self.shapes = []
        self.imgToAnns, self.catToImgs = defaultdict(list), defaultdict(list)
        self.idxToImg = []
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

        if meta_file is not None:
            MULTIPLE_OF = 32
            if os.path.isfile(meta_file):
                read_dict = pandas.read_csv(meta_file).to_dict()
            else:
                self.imgToMeta = None
            return_dict = {}
            for number in read_dict['image_name']:
                id = int(read_dict['image_name'][number].split('.')[0])
                try:
                    height = float(read_dict['altitude_normalized'][number])
                except:
                    height = float(read_dict['altitude(feet)'][number])

                return_dict[id] = int(height / MULTIPLE_OF) * MULTIPLE_OF

            self.imgToMeta = return_dict

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
                if 'name' in self.dataset['categories'][0].keys():
                    return {int(category['id']): category['name'] for category in self.dataset['categories']}

                return {int(category['id']): category for category in self.dataset['categories']}
            else:
                return {_id: category for _id, category in enumerate(self.dataset['categories'])}

        if isinstance(self.dataset['categories'], dict):
            return {int(k): v for k, v in self.dataset['categories'].items()}

    def _createIndex(self):
        # create index
        print('# Creating index...')
        anns, imgs = {}, {}
        imgToAnns = defaultdict(list)
        shapes = []

        for img in self.dataset['images']:
            img = self._shrink_img_entry_to_max_size(img)
            img_id = img['id']
            imgs[img_id] = img
            imgToAnns[img_id] = []
            shapes.append([img['width'], img['height']])

        for ann in tqdm.tqdm(self.dataset['annotations'], "Create annotations"):
            try:
                img = imgs[ann['image_id']]
            except KeyError:
                continue
            whole_img_shrink_scale = img['whole_shrink_scale']

            anns[ann['id']] = {
                'x': ann['bbox'][0] * whole_img_shrink_scale,
                'y': ann['bbox'][1] * whole_img_shrink_scale,
                'width': ann['bbox'][2] * whole_img_shrink_scale,
                'height': ann['bbox'][3] * whole_img_shrink_scale,
                'class_id': ann['category_id']
            }
            imgToAnns[ann['image_id']].append(anns[ann['id']])

        # throw all images without annotation away
        for img_idx in list(imgs.keys()):
            if len(imgToAnns[img_idx]) == 0:
                imgs.pop(img_idx)

        self.n_c = len(self.dataset['categories'])

        self.cats = self._parse_categories()

        print('# Index created!')

        # create class members
        self.anns = anns
        self.imgs = imgs
        self.shapes = np.array(shapes)
        self.imgToAnns = imgToAnns
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

    def info(self):
        """
        Print information about the annotation file.
        :return:
        """
        for key, value in self.dataset['info'].items():
            print('{}: {}'.format(key, value))

    def load_annotations(self, image_index):
        # get ground truth annotations
        annotations = np.zeros((0, 5))

        for idx, a in enumerate(self.imgToAnns[image_index]):

            # some annotations have basically no width / height, skip them
            if a['width'] < 1 or a['height'] < 1:
                continue

            annotation = np.zeros((1, 5))
            annotation[0, :4] = [a['x'], a['y'], (a['x'] + a['width']), (a['y'] + a['height'])]
            annotation[0, 4] = a['class_id']
            annotations = np.append(annotations, annotation, axis=0)

        return annotations

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
            path = self.imgs[img_idx]['file_name']
            img = cv2.imread(os.path.join(self.root, path))
            metadata = self.imgToMeta[img_idx]

            if img is None:
                raise Exception("Could not load img: %s" % path)

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        sample = {'img': torch.from_numpy(img), 'annot': torch.from_numpy(target), 'img_id': img_idx, 'meta': {
            'max_objs_in_image': self.max_objs_in_image
        }, 'metadata': metadata}
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


def convert_to_coco_api(ds):
    coco_ds = COCO()
    # annotation IDs need to start at 1, not 0, see torchvision issue #1530
    ann_id = 1
    dataset = {'images': [], 'categories': [], 'annotations': []}
    categories = set()
    for img_idx in range(len(ds)):
        # find better way to get target
        # targets = ds.get_annotations(img_idx)
        batch = ds[img_idx]
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


class PrepareBatch(object):
    def __init__(self):
        pass

    def __call__(self, sample):

        if not isinstance(sample['img'], torch.Tensor):
            sample['img'] = torch.from_numpy(sample['img'])

        if not isinstance(sample['annot'], torch.Tensor):
            sample['annot'] = torch.from_numpy(sample['annot'])

        return sample


if __name__ == '__main__':
    data_path = '/data/datasets/vis_drone'

    PATHS = {
        "train": (os.path.join(data_path, "images", "train"),
                  os.path.join(data_path, "annotations", 'instances_train.json')),
        "val": (os.path.join(data_path, "images", "val"),
                os.path.join(data_path, "annotations", 'instances_val.json')),
    }

    img_folder = PATHS['train'][0]
    ann_file = PATHS['train'][1]

    from data_loader.augmentation import RandomCropping
    from torchvision import transforms
    from data_loader.utils import AnnotCountRatio

    dataset = FlightDetectionDataset(img_folder, ann_file, transform=transforms.Compose([AnnotCountRatio(10)]),
                                     max_whole_image_size=4096, cache_images_in_memory=False)

    import matplotlib.pyplot as plt
    import data_loader.bounding_box_util as bb
    import tqdm

    for i in tqdm.tqdm(range(1000)):
        sample = dataset[np.random.randint(0, len(dataset))]
        path_name = dataset.imgs[sample['img_id']]['file_name']

        img = sample['img']

        for an in sample['annot']:
            img = bb.add_bbox_xyxy(img, an[0], an[1], an[2], an[3])

        plt.figure()
        plt.imshow(img)
        plt.title(path_name)
        plt.show()
        # plt.imsave('/tmp/test/%i.png' % i, img)