from efficientdet.train import EfficientDetModule, FlightDetectionDataset, PrepareBatch, SplitMode
import data_loader.bounding_box_util as bb_util


import argparse
import torch
from torchvision import transforms
import os
from efficientdet.src.dataset import ResizerRectangle, Normalizer, collater
import time
import matplotlib.pyplot as plt
import tqdm


def store_predicted_bbox(output_folder, _img, _gt_anns, _pred, _name, confident_threshold):
    _img = bb_util.to_numpy_array(_img)

    _img_gt = _img.copy()
    for _gt_ann in _gt_anns:
        if _gt_ann[4] > -1:
            _img_gt = bb_util.add_bbox_xyxy(_img_gt, _gt_ann[0], _gt_ann[1], _gt_ann[2], _gt_ann[3],
                                         label='Ground Truth (%i)' % _gt_ann[4])

    plt.imsave(os.path.join(os.path.join(output_folder, 'gt'), _name + "_gt.png"), _img_gt)

    _labels, _probs, _bboxes = _pred['labels'], _pred['probs'], _pred['boxes']
    for (_label, _prob, _bbox) in zip(_labels, _probs, _bboxes):
        if _prob > confident_threshold:
            width = _bbox[2]
            height = _bbox[3]
            if width * height < 50:
                # too small for a label
                _img = bb_util.add_bbox_xywh(_img, _bbox[0], _bbox[1], width, height)
            else:
                _img = bb_util.add_bbox_xywh(_img, _bbox[0], _bbox[1], width, height,
                                             label='Predicted (%i with %f%%)'
                                                   % (_label,
                                                      _prob))

    plt.imsave(os.path.join(os.path.join(output_folder, 'pred'), _name + "_pred.png"), _img)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--image_size", default=512)
    parser.add_argument("--output", default='.')
    opt = parser.parse_args()

    model = EfficientDetModule.load_from_checkpoint(opt.checkpoint, opt)
    model.eval()

    PATHS = {
        "train": (os.path.join(opt.data_path, "images", "train"),
                  os.path.join(opt.data_path, "annotations", 'instances_train.json')),
        "val": (os.path.join(opt.data_path, "images", "val"),
                os.path.join(opt.data_path, "annotations", 'instances_val.json')),
    }

    img_folder, ann_file = PATHS['train']
    dataset = FlightDetectionDataset(img_folder, ann_file, transform=transforms.Compose(
        [PrepareBatch(), Normalizer(), ResizerRectangle(opt.image_size)]), split_mode=SplitMode.SQUARE,
                                               remove_empty_imgs=True)

    inference_time = 0
    whole_time = 0

    with torch.no_grad():
        for i in tqdm.tqdm(range(len(dataset))):
            start_time = time.time()
            sample = collater([dataset[i]])
            img, ann = sample['img'][0], sample['annot'][0]
            start_time_inference = time.time()
            s, l, b = model(img.unsqueeze(0).float())
            stop_time_inference = time.time()
            _pred = {
                'boxes': b.cpu(),
                'labels': l.cpu(),
                'probs': s.cpu()
            }
            store_predicted_bbox(opt.output, img.cpu(), ann.cpu(), _pred, "%i" % i, 0.7)
            stop_time = time.time()

            inference_time += stop_time_inference - start_time_inference
            whole_time += stop_time - start_time

    print("Inference per object: %f" % (inference_time / len(dataset)))
    print("Time per object: %f" % (whole_time / len(dataset)))






