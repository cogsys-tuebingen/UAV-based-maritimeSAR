import os
import argparse
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
import random
from torch.utils.data import DataLoader
from evaluation.cocoeval import COCOeval
from tidecv import TIDE, datasets
import tqdm
import itertools

from data_loader.flight_detection_dataset import PrepareBatch, convert_to_coco_api
import data_loader.bounding_box_util as bb_util
from efficientdet.train import EfficientDetModule, ResizerRectangle, Normalizer, collater, FitIntoGrid
from data_loader.flight_detection_dataset import FlightDetectionDataset
from utils.run_utils import get_current_git_hash, get_slurm_job_id
from data_loader.dataset_configuration import load_dataset_configuration
from utils.tide_util import coco_dt_to_tide_data, coco_gt_to_tide_data


def get_args():
    parser = argparse.ArgumentParser(
        "EfficientDet: Scalable and Efficient Object Detection implementation by Signatrix GmbH")
    parser.add_argument("--batch_size", type=int, default=16, help="The number of images per batch")
    parser.add_argument("--data_path", type=str, required=True, help="the root folder of dataset")
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dataset_name", type=str, required=True, help="dataset name")
    parser.add_argument("--img_size", type=int, required=True)

    args = parser.parse_args()
    return args


def show_predicted_bbox(_img, _gt_anns, _pred, confident_threshold):
    _img = bb_util.to_numpy_array(_img)

    for _gt_ann in _gt_anns:
        if _gt_ann[4] > -1:
            _img = bb_util.add_bbox_xyxy(_img, _gt_ann[0], _gt_ann[1], _gt_ann[2], _gt_ann[3],
                                         label='Ground Truth (%i)' % _gt_ann[4])

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

    plt.figure()
    plt.imshow(_img)
    plt.show()


def load_checkpoint(path):
    model = EfficientDetModule.load_from_checkpoint(path)
    return model


@torch.no_grad()
def predict_whole_valset(model, val_generator, device):
    coco_pred = []
    coco_pred_img_ids = []

    for sample in tqdm.tqdm(val_generator, desc='Predict validation set (whole)'):
        # calc for map
        scores_batch, labels_batch, boxes_batch = [], [], []
        for img in sample['img']:
            s, l, b = model(img.unsqueeze(0).float().to(device))
            scores_batch.append(s)
            labels_batch.append(l)
            boxes_batch.append(b)

            if len(boxes_batch) > 0:
                for i, (boxes, scale, scores, labels, img_id) in enumerate(zip(boxes_batch, sample['scale'],
                                                                               scores_batch, labels_batch,
                                                                               sample['img_id'])):
                    if len(boxes) == 0:
                        continue

                    boxes_batch[i][:, 2] -= boxes_batch[i][:, 0]  # to width
                    boxes_batch[i][:, 3] -= boxes_batch[i][:, 1]  # and height

                    for box_id in range(boxes.shape[0]):
                        score = float(scores[box_id])
                        label = int(labels[box_id])
                        box = boxes[box_id, :]

                        if score < model.hparams['score_threshold']:
                            continue

                        image_result = {
                            'image_id': img_id,
                            'category_id': label,
                            'score': float(score),
                            'bbox': box.tolist(),
                        }

                        coco_pred.append(image_result)
                    coco_pred_img_ids.append(img_id)

    return coco_pred_img_ids, coco_pred


def print_result(type, network_name, backbone, img_size, coco_stats, tide_stats, random_cropping):
    url = input("Url of the run:")

    s = """
    ***
    
    {{
        "network": "{network}",
        "backbone": "{backbone}",
        "type": "{type}",
        "random_cropping": {random_cropping},
        "img_size": {image_size},
        "reference": "{url}",
        "val": {{
          "mAP": {{
             "Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ]":  {map},
             "Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ]":  {map_05},
             "Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ]":  {map_075},
             "Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ]":  {map_s},
             "Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ]":  {map_m},
             "Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ]":  {map_l},
             "Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ]":  {mar_1},
             "Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ]":  {mar_10},
             "Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ]":  {mar_100},
             "Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ]":  {mar_100_s},
             "Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ]":  {mar_100_m},
             "Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ]":  {mar_100_l}
          }},
          "TIDE": {{
            "Cls": {tide_cls},
            "Loc": {tide_loc},
            "Both": {tide_both},
            "Dupe": {tide_dupe},
            "Bkg": {tide_bkg},
            "Miss": {tide_miss},
            "FalsePos": {tide_fp},
            "FalseNeg": {tide_fn}
          }}
        }}
      }}""".format(map=coco_stats[0], map_05=coco_stats[1], map_075=coco_stats[2], map_s=coco_stats[3],
                   map_m=coco_stats[4],
                   map_l=coco_stats[5], mar_1=coco_stats[6], mar_10=coco_stats[7], mar_100=coco_stats[8],
                   mar_100_s=coco_stats[9], mar_100_m=coco_stats[10], mar_100_l=coco_stats[11],
                   tide_cls=tide_stats['main']['default']['Cls'],
                   tide_loc=tide_stats['main']['default']['Loc'],
                   tide_both=tide_stats['main']['default']['Both'],
                   tide_dupe=tide_stats['main']['default']['Dupe'],
                   tide_bkg=tide_stats['main']['default']['Bkg'],
                   tide_miss=tide_stats['main']['default']['Miss'],
                   tide_fp=tide_stats['special']['default']['FalsePos'],
                   tide_fn=tide_stats['special']['default']['FalseNeg'],
                   url=url, network=network_name, backbone=backbone, image_size=img_size, type=type,
                   random_cropping="true" if random_cropping else "false"
                   )

    print(s)


if __name__ == "__main__":
    opt = get_args()
    num_gpus = torch.cuda.device_count()
    on_cluster = num_gpus > 1
    device = 'cuda'

    # fix the seed for reproducibility
    seed = opt.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    hparams = vars(opt)
    hparams['number_workers'] = 0

    model = load_checkpoint(opt.checkpoint_path).to(device)
    model.eval()

    print("Opt: %s" % opt)
    print("Hparams: %s" % model.hparams)

    PATHS = {
        "train": (os.path.join(opt.data_path, "images", "train"),
                  os.path.join(opt.data_path, "annotations", 'instances_train.json')),
        "val": (os.path.join(opt.data_path, "images", "val"),
                os.path.join(opt.data_path, "annotations", 'instances_val.json')),
    }

    img_folder, ann_file = PATHS['val']

    with torch.no_grad():
        merged_prediction = False

        val_set = FlightDetectionDataset(img_folder, ann_file, opt.img_size, transform=transforms.Compose(
            [PrepareBatch(), Normalizer(), ResizerRectangle(model.hparams['image_size']), FitIntoGrid(128)]))
        coco_gt = convert_to_coco_api(val_set)

        val_params = {"batch_size": 1,
                      "shuffle": False,
                      "drop_last": False,
                      "collate_fn": collater,
                      "num_workers": hparams['number_workers'],
                      "pin_memory": True}

        val_generator = DataLoader(val_set, **val_params)

        coco_pred_img_ids, coco_pred = predict_whole_valset(model, val_generator, device)
        coco_dt = coco_gt.loadRes(coco_pred)

        print("Calculate mAP")
        coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
        coco_eval.params.imgIds = coco_pred_img_ids
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        coco_stats = coco_eval.stats

        print("Calculate TIDE")
        tide = TIDE()
        tide.evaluate(coco_gt_to_tide_data(coco_gt),
                      coco_dt_to_tide_data(coco_dt),
                      mode=TIDE.BOX)
        tide.summarize()
        tide.plot()
        tide_stats = tide.get_all_errors()

        network = model.hparams['network'] if 'network' in model.hparams.keys() else '?'
        backbone = model.hparams['backbone'] if 'backbone' in model.hparams.keys() else model.hparams[
            'efficientnet_model']
        random_cropping = model.hparams['random_cropping'] if 'random_cropping' in model.hparams.keys() else False

        print_result(model.hparams['network'], backbone, model.hparams['image_size'], coco_stats, tide_stats, random_cropping)

        # visualize prediction
        for img_id in coco_pred_img_ids:
            dataset_idx = val_set.idxToImg.index(img_id)
            sample = val_set[dataset_idx]
            preds = [pred for pred in coco_pred if pred['image_id'] == img_id]
            bboxes = [p['bbox'] for p in preds]
            probs = [p['score'] for p in preds]
            labels = [p['category_id'] for p in preds]

            pred = {
                'boxes': bboxes,
                'labels': labels,
                'probs': probs
            }

            show_predicted_bbox(sample['img'], sample['annot'], pred, 0.5)

        del coco_pred

    with torch.no_grad():
        idx = np.random.choice(list(range(len(val_set))), 10)
        for i in idx:
            sample = val_set[i]
            img = sample['img']
            s, l, b = model(img.permute(2, 0, 1).unsqueeze(0).float().to(device))

            b[:, 2] -= b[:, 0]  # to width
            b[:, 3] -= b[:, 1]  # and height

            pred = {
                'boxes': b.cpu(),
                'labels': l.cpu(),
                'probs': s.cpu()
            }

            show_predicted_bbox(sample['img'], sample['annot'], pred, 0.7)
            del sample
