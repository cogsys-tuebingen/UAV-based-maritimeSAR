from tidecv import Data, f
from pycocotools.coco import COCO


def coco_gt_to_tide_data(coco_gt: COCO, name="default"):
    data = Data(name)

    for idx, image in enumerate(coco_gt.imgs.values()):
        if 'file_name' in image.keys():
            data.add_image(image['id'], image['file_name'])

    if coco_gt.cats is not None:
        for cat in coco_gt.cats.values():
            if 'name' in cat.keys():
                data.add_class(cat['id'], cat['name'])

    for ann in coco_gt.anns.values():
        image = ann['image_id']
        _cls = ann['category_id']
        box = ann['bbox']

        if ann['iscrowd']:
            data.add_ignore_region(image, _cls, box)
        else:
            data.add_ground_truth(image, _cls, box)

    return data


def coco_dt_to_tide_data(coco_dt: COCO, name="default"):
    data = Data(name)

    for det in coco_dt.anns.values():
        image = det['image_id']
        _cls = det['category_id']
        score = det['score']
        box = det['bbox'] if 'bbox' in det else None
        mask = det['segmentation'] if 'segmentation' in det else None

        data.add_detection(image, _cls, score, box, mask)

    return data