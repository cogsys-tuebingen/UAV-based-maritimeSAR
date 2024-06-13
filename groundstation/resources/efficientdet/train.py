import os
import argparse
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import pytorch_lightning as lightning
from pytorch_lightning.loggers.wandb import WandbLogger
import wandb
from itertools import chain
from evaluation.cocoeval import COCOeval
import numpy as np
import random
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from efficientdet.src.dataset import collater
from data_loader.utils import FitIntoGrid, ResizerRectangle, Normalizer
from data_loader.augmentation import RandomFlipping
from data_loader.flight_detection_dataset import PrepareBatch, convert_to_coco_api
from data_loader.flight_detection_dataset import FlightDetectionDataset
from efficientdet.src.model import EfficientDet, Anchors
import data_loader.bounding_box_util as bb_util
from efficientdet.src.loss import FocalLoss
from utils.run_utils import get_current_git_hash, get_slurm_job_id, get_pip_packages, get_wandb_log_dir
from utils.pytorch_lightning_hacks import MyDDP
from data_loader.dataset_configuration import load_dataset_configuration
from utils.optimize_anchors import optimize_anchors
from utils.arg_parser_util import str2config_variant

from data_loader.augmentation import RandomCropping
from lightning_callbacks.lr_logger import LRLoggingCallback


def load_dataset(data_path, type, image_size, random_cropping):
    PATHS = {
        "train": (os.path.join(data_path, "images", "train"),
                  os.path.join(data_path, "annotations", 'instances_train.json')),
        "val": (os.path.join(data_path, "images", "val"),
                os.path.join(data_path, "annotations", 'instances_val.json')),
    }

    img_folder, ann_file = PATHS[type]

    if type == 'train':
        preprocessing = [PrepareBatch(), RandomFlipping(), Normalizer()]
        if random_cropping:
            preprocessing.append(RandomCropping(0.5))
        preprocessing += [ResizerRectangle(image_size), FitIntoGrid(128)]

        return FlightDetectionDataset(img_folder, ann_file, image_size, transform=transforms.Compose(preprocessing))
    elif type == 'val':
        return FlightDetectionDataset(img_folder, ann_file, image_size, transform=transforms.Compose(
            [PrepareBatch(), Normalizer(), ResizerRectangle(image_size), FitIntoGrid(128)]))


class EfficientDetModule(lightning.LightningModule):
    def __init__(self, hparams):
        super(EfficientDetModule, self).__init__()

        self.hparams = hparams
        self.batch_size = hparams['batch_size']  # used for the autoscaling algorithm
        self.save_hyperparameters()

        self.criterion = FocalLoss(self._get_hparam('thres_found_object_iou'),
                                   self._get_hparam('thres_not_found_object_iou')).cpu()
        self.anchors = Anchors(scales=self.hparams["anchor_scales"])
        self.model = EfficientDet(self.hparams, self.anchors, num_classes=self.hparams['num_classes'],
                                  efficientnet_model=self.hparams['efficientnet_model'])
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
        if stage == 'fit':
            self.training_set = load_dataset(self.hparams['data_path'], 'train',
                                             self.hparams['image_size'],
                                             random_cropping=hparams['random_cropping'])
            self.val_set = load_dataset(self.hparams['data_path'], 'val',
                                        self.hparams['image_size'],
                                        random_cropping=False)

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

        classification, regression, anchors = self.model(batch['img'], is_training=True)
        cls_loss, reg_loss = self.criterion(classification, regression, anchors, batch['annot'])
        loss = cls_loss.mean() + reg_loss.mean()

        logs = {
            "train/loss": loss.item(),
            "train/cls_loss": cls_loss.item(),
            "train/reg_loss": reg_loss.item(),
        }

        for k, v in logs.items():
            self.log(k, v, sync_dist=True)

        return {
            'loss': loss,
            "meta/seen_objects": (batch['annot'][:, :, 4] != -1).sum().item()
        }

    def training_epoch_end(self, outputs):
        self.count_of_seen_objects += np.sum([o['meta/seen_objects'] for o in outputs])

        self.log("meta/seen_objects", self.count_of_seen_objects)

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        classification, regression, anchors = self.model(batch['img'], is_training=True)
        cls_loss, reg_loss = self.criterion(classification, regression, anchors, batch['annot'])
        loss = cls_loss.mean() + reg_loss.mean()

        # TODO: fix two forward passes

        # calc for map
        scores_batch, labels_batch, boxes_batch = [], [], []
        for img in batch['img']:
            s, l, b = self.model(img.unsqueeze(0), is_training=False)
            scores_batch.append(s)
            labels_batch.append(l)
            boxes_batch.append(b)

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

        return {'val_loss': loss,
                'val_cls_loss': cls_loss,
                'val_reg_loss': reg_loss,
                'val_pred_results': results,
                'val_img_ids': batch['img_id']}

    def validation_epoch_end(self, outputs):
        loss = torch.stack([o['val_loss'] for o in outputs]).mean().detach()
        cls_loss = torch.stack([o['val_cls_loss'] for o in outputs]).mean().detach()
        reg_loss = torch.stack([o['val_reg_loss'] for o in outputs]).mean().detach()

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
            "val/cls_loss": cls_loss,
            "val/reg_loss": reg_loss,
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

        self.log("val_loss", loss)
        self.log("val_map", averaged_stats[0])

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

        _img = bb_util.to_numpy_array(_img)

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

        img = wandb.Image(bb_util.to_numpy_array(_img), boxes=box_data)
        return img

    def forward(self, x, is_training=False):
        return self.model(x, is_training)


def get_args():
    parser = argparse.ArgumentParser(
        "EfficientDet: Scalable and Efficient Object Detection implementation by Signatrix GmbH")
    parser.add_argument("--image_size", type=int, default=None, help="The common width and height for all images")
    parser.add_argument("--batch_size", type=int, default=None, help="The number of images per batch")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument('--alpha', type=float, default=0.25)
    parser.add_argument('--gamma', type=float, default=1.5)
    parser.add_argument("--num_epochs", type=int, default=500)
    parser.add_argument("--test_interval", type=int, default=1, help="Number of epoches between testing phases")
    parser.add_argument("--es_min_delta", type=float, default=0.0,
                        help="Early stopping's parameter: minimum change loss to qualify as an improvement")
    parser.add_argument("--es_patience", type=int, default=0,
                        help="Early stopping's parameter: number of epochs with no improvement after which training will be stopped. Set to 0 to disable this technique.")
    parser.add_argument("--data_path", type=str, required=True, help="the root folder of dataset")
    parser.add_argument("--dataset_name", type=str, required=True, help="dataset name")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log_path", type=str, default=None)
    parser.add_argument("--optimize_anchors", action='store_true', default=False)
    parser.add_argument("--online_logging", action='store_true', default=False)
    parser.add_argument("--config_variant", default=None, type=str2config_variant)
    parser.add_argument("--random_cropping", action='store_true', default=False)
    parser.add_argument("--checkpoint_path", default=None)
    parser.add_argument("--server_name", default=None, help="Is used to determine the correct settings [None, avalon1]")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
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
    hparams['network'] = 'efficientdet'
    hparams['type'] = 'whole'
    hparams['dataset'] = opt.dataset_name

    hparams['score_threshold'] = 0.1
    hparams['batch_size'] = opt.batch_size
    hparams['train_with_batchnorm'] = True
    hparams['floating16'] = False

    hparams['number_workers'] = 8

    # load the dataset config to the configuration
    hparams = {**hparams, **load_dataset_configuration(hparams['network'], hparams['dataset'],
                                                       hparams['type'], hparams['config_variant'],
                                                       server_name=hparams['server_name'])}

    if opt.optimize_anchors:
        print("# Optimize the anchor scales for the dataset")
        hparams['anchor_scales'] = optimize_anchors(load_dataset(opt.data_path, 'train', opt.image_size))

    if opt.log_path is None:
        opt.log_path = get_wandb_log_dir()

    model = EfficientDetModule(hparams)

    print("HParams: %s" % hparams)
    print("Installed packages: %s" % get_pip_packages())

    logger = WandbLogger(offline=not opt.online_logging,
                         save_dir=opt.log_path, project='crow-efficientdet')

    checkpoint_callback = lightning.callbacks.ModelCheckpoint(
        save_top_k=1,
        verbose=True,
        monitor='val_map',
        mode='max',
        prefix='',
        save_last=True,
    )

    trainer = lightning.Trainer(default_root_dir=opt.log_path,
                                max_epochs=opt.num_epochs, min_epochs=min(50, opt.num_epochs),
                                gpus=-1, logger=logger,
                                distributed_backend=backend, gradient_clip_val=0.1,
                                checkpoint_callback=True,
                                callbacks=[LRLoggingCallback(), EarlyStopping(monitor="val_loss", mode="min"),
                                           checkpoint_callback],
                                plugins=[MyDDP()])

    trainer.fit(model)
