import os
import argparse

import torch
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import transforms
import pytorch_lightning as lightning
from pytorch_lightning.loggers.wandb import WandbLogger
import wandb
from itertools import chain
from evaluation.cocoeval import COCOeval
import numpy as np
import random
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from faster_rcnn.model import fasterrcnn_resnet_fpn
from faster_rcnn.dataset import collater

from data_loader.utils import FitIntoGrid, ResizerRectangle, Normalizer
from data_loader.augmentation import RandomFlipping
from data_loader.flight_detection_dataset import PrepareBatch, convert_to_coco_api
from data_loader.flight_detection_dataset import FlightDetectionDataset
from data_loader.flight_detection_dataset_multispectral import FlightDetectionDatasetMultispectral

import data_loader.bounding_box_util as bb_util
from utils.run_utils import get_current_git_hash, get_slurm_job_id, get_pip_packages, get_wandb_log_dir, \
    get_slurm_job_path
from utils.pytorch_lightning_hacks import increase_filedesc_limit, MyDDP
from data_loader.dataset_configuration import load_dataset_configuration
from data_loader.augmentation import RandomCropping
from cp_evaluation.preprocessor_factory import generate_preprocessing_pipelines

from utils.arg_parser_util import str2config_variant
from lightning_callbacks.lr_logger import LRLoggingCallback


def load_dataset(data_path, type, image_size, random_cropping, preprocessing_pipeline, multispectral, use_crow):
    PATHS = {
        "train": (os.path.join(data_path, "images", "train"),
                  os.path.join(data_path, "annotations", 'instances_train.json')),
        "val": (os.path.join(data_path, "images", "val"),
                os.path.join(data_path, "annotations", 'instances_val.json')),
    }

    img_folder, ann_file = PATHS[type]

    if multispectral:
        dataset_class = FlightDetectionDatasetMultispectral
    else:
        dataset_class = FlightDetectionDataset

    preprocessing = [PrepareBatch()]
    preprocessing += preprocessing_pipeline

    if type == 'train':
        if random_cropping:
            preprocessing.append(RandomCropping(0.5))
        preprocessing += [RandomFlipping(), Normalizer(), ResizerRectangle(image_size), FitIntoGrid(128)]

        dataset = dataset_class(img_folder, ann_file, image_size, transform=transforms.Compose(preprocessing))

        if use_crow:
            from data_loader.flight_detection_dataset_splitter_v3 import FlightDetectionDataset as CrowDataset
            dataset_crow = CrowDataset(img_folder, ann_file, image_size, transform=transforms.Compose(preprocessing))

            dataset = ConcatDataset([dataset, dataset_crow])

        return dataset
    elif type == 'val':
        preprocessing += [Normalizer(), ResizerRectangle(image_size), FitIntoGrid(128)]
        return dataset_class(img_folder, ann_file, image_size, transform=transforms.Compose(
            preprocessing))


class FasterRCNNModule(lightning.LightningModule):
    def __init__(self, hparams):
        super(FasterRCNNModule, self).__init__()

        self.save_hyperparameters(hparams)
        self.batch_size = hparams['batch_size']  # used for the autoscaling algorithm

        # backward compatibility
        self.hparams['input_channel_count'] = self.hparams[
            'input_channel_count'] if 'input_channel_count' in self.hparams.keys() else 3

        self.model = fasterrcnn_resnet_fpn(pretrained=False, progress=True,
                                           num_classes=self._get_hparam('num_classes'),
                                           backbone_arch=self._get_hparam('backbone'),
                                           pretrained_backbone=self._get_hparam('pretrained_backbone'),
                                           box_fg_iou_thresh=self._get_hparam('thres_found_object_iou'),
                                           box_bg_iou_thresh=self._get_hparam('thres_not_found_object_iou'),
                                           box_score_thresh=self._get_hparam('score_threshold'),
                                           box_nms_thresh=0.5)
        if self.hparams['input_channel_count'] != 3:
            org_conv = self.model.backbone.body.conv1
            self.model.backbone.body.conv1 = torch.nn.Conv2d(self.hparams['input_channel_count'], org_conv.out_channels,
                                                             kernel_size=org_conv.kernel_size, stride=org_conv.stride,
                                                             padding=org_conv.padding, dilation=org_conv.dilation,
                                                             groups=org_conv.groups,
                                                             bias=org_conv.bias, padding_mode=org_conv.padding_mode)
            trainable_layers = 5
            layers_to_train = ['layer4', 'layer3', 'layer2', 'layer1', 'conv1'][:trainable_layers]
            if trainable_layers == 5:
                layers_to_train.append('bn1')
            for name, parameter in self.model.backbone.named_parameters():
                if all([name.startswith(layer) for layer in layers_to_train]):
                    parameter.requires_grad_(True)

        self.count_of_seen_objects = 0

    def _get_hparam(self, key):
        """
            default value: None
        """

        if key in self.hparams.keys():
            return self.hparams[key]
        else:
            print("? %s is not defined -> use default value" % key)
            return None

    def setup(self, stage: str):
        train_pipeline, validation_pipeline, test_pipeline = generate_preprocessing_pipelines(
            self.hparams['preprocessing_train'],
            self.hparams['preprocessing_val'],
            self.hparams['preprocessing_test'])

        if stage == 'fit':
            self.training_set = load_dataset(self.hparams['data_path'], 'train',
                                             self.hparams['image_size'],
                                             random_cropping=hparams['random_cropping'],
                                             preprocessing_pipeline=train_pipeline,
                                             multispectral=self.hparams['multispectral_dataset'],
                                             use_crow=self.hparams['crow'])
            self.val_set = load_dataset(self.hparams['data_path'], 'val',
                                        self.hparams['image_size'],
                                        random_cropping=False,
                                        preprocessing_pipeline=validation_pipeline,
                                        multispectral=self.hparams['multispectral_dataset'],
                                        use_crow = False)

            # used for the mAP calculation
            self.val_set_coco = None
            self.val_set_coco = convert_to_coco_api(self.val_set)

    def train_dataloader(self) -> DataLoader:
        training_params = {"batch_size": self.batch_size,
                           "shuffle": True,
                           "drop_last": True,
                           "collate_fn": collater,
                           "num_workers": self.hparams['number_workers']}

        training_generator = DataLoader(self.training_set, **training_params)
        return training_generator

    def val_dataloader(self) -> DataLoader:
        test_params = {"batch_size": self.batch_size,
                       "shuffle": False,
                       "drop_last": False,
                       "collate_fn": collater,
                       "num_workers": self.hparams['number_workers']}

        test_generator = DataLoader(self.val_set, **test_params)
        return test_generator

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), self.hparams['lr'])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss"
        }

    def training_step(self, batch, batch_idx):
        if not self._get_hparam('train_with_batchnorm'):
            self.model.freeze_bn()

        targets = [{'boxes': b[..., :4], 'labels': b[..., -1].type(torch.int64)} for b in batch['annot']]
        loss_dict, _ = self.model(batch['img'], targets)

        losses = sum(loss for loss in loss_dict.values())
        logs = {
            "train/proposal_losses/objectness_loss": loss_dict["loss_objectness"],
            "train/proposal_losses/rpn_box_reg_loss": loss_dict["loss_rpn_box_reg"],
            "train/detector_losses/classifier_loss": loss_dict["loss_classifier"],
            "train/detector_losses/box_reg_loss": loss_dict["loss_box_reg"]
        }

        for k, v in logs.items():
            self.log(k, v, sync_dist=True)

        return {
            'loss': losses,
            "meta/seen_objects": len(batch['annot'])
        }

    def training_epoch_end(self, outputs):
        self.count_of_seen_objects += np.sum([o['meta/seen_objects'] for o in outputs])

        self.log("meta/seen_objects", self.count_of_seen_objects)

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        targets = [{'boxes': b[..., :4], 'labels': b[..., -1].type(torch.int64)} for b in batch['annot']]

        loss_dict, detections = self.model(batch['img'], targets)
        detections = {k: [d[k] for d in detections] for k in detections[0].keys()}
        boxes, labels, scores = detections['boxes'], detections['labels'], detections['scores']

        losses = sum(loss for loss in loss_dict.values())

        scores_batch, labels_batch, boxes_batch = [], [], []

        boxes_batch = boxes
        labels_batch = labels
        scores_batch = scores

        results = []

        if len(boxes_batch) > 0:
            for i, (boxes, scores, labels, img_id) in enumerate(zip(boxes_batch,
                                                                    scores_batch, labels_batch,
                                                                    batch['img_id'])):
                if len(boxes) == 0:
                    continue

                boxes_batch[i][:, 2] -= boxes_batch[i][:, 0]  # to width
                boxes_batch[i][:, 3] -= boxes_batch[i][:, 1]  # and height

                for box_id in range(boxes.shape[0]):
                    score = float(scores[box_id])
                    label = int(labels[box_id])
                    box = boxes[box_id, :]

                    if score < self.hparams['score_threshold']:
                        continue

                    image_result = {
                        'image_id': img_id,
                        'category_id': label,
                        'score': float(score),
                        'bbox': box.tolist(),
                    }

                    results.append(image_result)

        if self.current_epoch % 10 == 0 and batch_idx < 3:
            previews = []
            for i in range(len(batch['img'])):
                _pred = {
                    'boxes': boxes_batch[i],
                    'labels': labels_batch[i],
                    'probs': scores_batch[i]
                }
                previews.append(self._generate_preview(batch['img'][i].cpu(), batch['annot'][i].cpu(),
                                                       _pred, self.val_set.cats))

            self.logger.experiment.log({'preview': previews})

        return {'val_loss': losses,
                "val_loss_objectness": loss_dict["loss_objectness"],
                "val_loss_rpn_box_reg": loss_dict["loss_rpn_box_reg"],
                "val_loss_classifier": loss_dict["loss_classifier"],
                "val_loss_box_reg": loss_dict["loss_box_reg"],
                'val_pred_results': results,
                'val_img_ids': batch['img_id']}

    def validation_epoch_end(self, outputs):
        loss = torch.stack([o['val_loss'] for o in outputs]).mean().detach()
        val_objectness_loss = torch.stack([o['val_loss_objectness'] for o in outputs]).mean().detach()
        val_rpn_reg_loss = torch.stack([o['val_loss_rpn_box_reg'] for o in outputs]).mean().detach()
        val_classifier_loss = torch.stack([o['val_loss_classifier'] for o in outputs]).mean().detach()
        val_box_reg_loss = torch.stack([o['val_loss_box_reg'] for o in outputs]).mean().detach()

        # merge the results for the mAP calc
        coco_pred = list(chain.from_iterable([o['val_pred_results'] for o in outputs]))
        if self.val_set_coco is not None and len(coco_pred) > 0:
            coco_true = self.val_set_coco
            coco_dt = coco_true.loadRes(coco_pred)
            image_ids = list(chain.from_iterable([o['val_img_ids'] for o in outputs]))

            print("Calculate mAP")
            coco_eval = COCOeval(coco_true, coco_dt, 'bbox')
            coco_eval.params.imgIds = image_ids
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()
            stats = coco_eval.stats
        else:
            stats = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float)

        map_all = self.all_gather(torch.tensor(stats).cuda())
        averaged_stats = map_all.mean(0)

        logs = {
            "val/loss": loss,
            "val/proposal_losses/objectness_loss": val_objectness_loss,
            "val/proposal_losses/rpn_box_reg_loss": val_rpn_reg_loss,
            "val/detector_losses/classifier_loss": val_classifier_loss,
            "val/detector_losses/box_head_loss": val_box_reg_loss,
        }

        for k, v in logs.items():
            self.log(k, v, on_epoch=True, sync_dist=True)

        # this is global logged, because the reduction would invalidate the results
        logs = {
            "Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=500 ]": averaged_stats[0],
            "Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=500 ]": averaged_stats[1],
            "Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=500 ]": averaged_stats[2],
            "Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=500 ]": averaged_stats[3],
            "Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=500 ]": averaged_stats[4],
            "Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=500 ]": averaged_stats[5],
            "Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ]": averaged_stats[6],
            "Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ]": averaged_stats[7],
            "Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=500 ]": averaged_stats[8],
            "Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=500 ]": averaged_stats[9],
            "Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=500 ]": averaged_stats[10],
            "Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=500 ]": averaged_stats[11]
        }
        self.logger.experiment.log(logs)

        self.log("val_loss", loss, batch_size=1)
        self.log("val_map", averaged_stats[0], batch_size=1)

        return {
            "val_los": loss,
            "val_map": averaged_stats[0]
        }

    def _generate_preview(self, _img, _gt_anns, _pred, categories):
        box_data = {
            "predictions": {
                "box_data": [],
                'class_labels': categories
            },
            "ground_truth": {
                "box_data": [],
                'class_labels': categories
            }
        }

        for _gt_ann in _gt_anns:
            if _gt_ann[4] > -1:
                box_data["ground_truth"]["box_data"].append({
                    "position": {
                        "minX": _gt_ann[0].item(),
                        "maxX": _gt_ann[2].item(),
                        "minY": _gt_ann[1].item(),
                        "maxY": _gt_ann[3].item(),
                    },
                    "class_id": int(_gt_ann[4].item()),
                    "domain": "pixel",
                    "box_caption": "gt (%i)" % int(_gt_ann[4].item()),
                    "scores": {
                        "score": 1.0,
                    }
                })

        _labels, _probs, _bboxes = _pred['labels'], _pred['probs'], _pred['boxes']
        for (_label, _prob, _bbox) in zip(_labels, _probs, _bboxes):
            width = _bbox[2].item()
            height = _bbox[3].item()
            box_data["predictions"]["box_data"].append({
                "position": {
                    "minX": _bbox[0].item(),
                    "maxX": _bbox[0].item() + width,
                    "minY": _bbox[1].item(),
                    "maxY": _bbox[1].item() + height,
                },
                "class_id": int(_label.item()),
                "box_caption": "pred (%i / %f)" % (int(_label.item()), _prob.item()),
                "domain": "pixel",
                "scores": {
                    "score": _prob.item(),
                }
            })

        if _img.shape[0] > 3:
            _img = _img[(2, 1, 0),]

        img = wandb.Image(bb_util.to_numpy_array(_img), boxes=box_data)
        return img

    def forward(self, x):
        return self.model(x)


def get_args():
    parser = argparse.ArgumentParser(
        "FasterRCNN")
    parser.add_argument("--image_size", type=int, default=None, help="The common width and height for all images")
    parser.add_argument("--batch_size", type=int, default=None, help="The number of images per batch")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument('--alpha', type=float, default=0.25)
    parser.add_argument('--gamma', type=float, default=1.5)
    parser.add_argument("--num_epochs", type=int, default=500)
    parser.add_argument("--test_interval", type=int, default=1, help="Number of epoches between testing phases")
    parser.add_argument("--es_min_delta", type=float, default=0.0,
                        help="Early stopping's parameter: minimum change loss to qualify as an improvement")
    parser.add_argument("--es_patience", type=int, default=10,
                        help="Early stopping's parameter: number of epochs with no improvement after which training will be stopped. Set to 0 to disable this technique.")
    parser.add_argument("--data_path", type=str, required=True, help="the root folder of dataset")
    parser.add_argument("--dataset_name", type=str, required=True, help="dataset name")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log_path", type=str, default=None)
    parser.add_argument("--online_logging", action='store_true', default=False)
    parser.add_argument("--config_variant", default=None, type=str2config_variant)
    parser.add_argument("--random_cropping", action='store_true', default=False)
    parser.add_argument("--checkpoint_path", default=None)
    parser.add_argument("--server_name", default=None, help="Is used to determine the correct settings [None, avalon1]")
    parser.add_argument("--project_name", default=None, type=str)
    parser.add_argument("--preprocessing_train", default=None, type=str)
    parser.add_argument("--preprocessing_val", default=None, type=str)
    parser.add_argument("--preprocessing_test", default=None, type=str)
    parser.add_argument("--backbone", default='resnet50', type=str,
                        help="valid backbones: ResNet, resnet18, resnet34, resnet50, resnet101, resnet152, resnext50_32x4d, \
                                resnext101_32x8d, wide_resnet50_2, wide_resnet101_2")
    parser.add_argument("--untrained", action="store_true")
    parser.add_argument("--multispectral_dataset", default=False, action='store_true')
    parser.add_argument("--crow", default=False, action='store_true')

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    increase_filedesc_limit()

    opt = get_args()
    num_gpus = torch.cuda.device_count()
    on_cluster = num_gpus > 1

    # fix the seed for reproducibility
    seed = opt.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    backend = 'ddp'

    if backend == 'dp':
        opt.batch_size *= num_gpus
        print("# Adapt batchsize with the gpu count: %i" % opt.batch_size)

    hparams = vars(opt)
    # store some run information
    hparams['git_id'] = get_current_git_hash()
    hparams['slurm_job_id'] = get_slurm_job_id()
    hparams['slurm_job_path'] = get_slurm_job_path()
    hparams['network'] = 'fasterrcnn'
    hparams['type'] = 'whole'
    hparams['dataset'] = opt.dataset_name

    hparams['input_channel_count'] = 3
    hparams['input_max_value'] = 255

    # for multispectral recordings
    if hparams['multispectral_dataset']:
        hparams['input_channel_count'] = 5
        hparams['input_max_value'] = pow(2, 12)

        if hparams['preprocessing_train'] is not None \
                and hparams['preprocessing_train'].startswith("colorspacereduction('rgb')"):
            assert hparams['preprocessing_val'] is None
            assert hparams['preprocessing_test'] is None
            hparams['input_channel_count'] = 3
            hparams['input_max_value'] = 255

    hparams['score_threshold'] = 0.1
    hparams['batch_size'] = opt.batch_size
    hparams['train_with_batchnorm'] = True
    hparams['floating16'] = False

    hparams['number_workers'] = 0

    hparams['pretrained_backbone'] = not opt.untrained
    hparams['backbone'] = opt.backbone

    # load the dataset config to the configuration
    overwritten_image_size = hparams['image_size']
    overwritten_batch_size = hparams['batch_size']
    hparams = {**hparams, **load_dataset_configuration(hparams['network'], hparams['dataset'],
                                                       hparams['type'], hparams['config_variant'],
                                                       server_name=hparams['server_name'])}
    hparams['image_size'] = hparams['image_size'] if overwritten_image_size is None else overwritten_image_size
    hparams['batch_size'] = hparams['batch_size'] if overwritten_batch_size is None else overwritten_batch_size

    if opt.log_path is None:
        opt.log_path = get_wandb_log_dir()

    model = FasterRCNNModule(hparams)

    print("HParams: %s" % hparams)
    print("Installed packages: %s" % get_pip_packages())

    logger = WandbLogger(offline=not opt.online_logging,
                         save_dir=opt.log_path, project=opt.project_name)

    checkpoint_callback = lightning.callbacks.ModelCheckpoint(
        save_top_k=1,
        verbose=True,
        monitor='val_map',
        mode='max',
        save_last=True,
    )

    trainer = lightning.Trainer(default_root_dir=opt.log_path,
                                max_epochs=opt.num_epochs, min_epochs=min(10, opt.num_epochs),
                                gpus=-1, logger=logger,
                                strategy=backend, gradient_clip_val=0.1,
                                checkpoint_callback=True,
                                precision=16 if hparams['floating16'] else 32,
                                callbacks=[LRLoggingCallback(), EarlyStopping(monitor="val_loss", mode="min",
                                                                              patience=opt.es_patience),
                                           checkpoint_callback]
                                )

    trainer.fit(model)
