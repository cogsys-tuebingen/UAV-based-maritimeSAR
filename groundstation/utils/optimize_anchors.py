import matplotlib.pyplot as plt
import os
import argparse
import numpy as np
import torch
from torchvision import transforms
import tqdm
import json

from data_loader.flight_detection_dataset import PrepareBatch
from data_loader.utils import ResizerSquare, ResizerRectangle, Normalizer, FitIntoGrid
from efficientdet.src.utils import Anchors, generate_anchors


def _get_anchors(scales=None):
    anchors_generator = Anchors(scales=scales)
    all_anchors = np.zeros((0, 2)).astype(np.float32)
    for idx, p in enumerate(anchors_generator.pyramid_levels):
        a = generate_anchors(base_size=anchors_generator.sizes[idx],
                             ratios=anchors_generator.ratios,
                             scales=anchors_generator.scales)

        all_anchors = np.append(all_anchors, a[:, 2:] - a[:, :2], axis=0)

    return all_anchors


def _get_bbox_shapes(dataset):
    shapes = np.zeros((0, 2))

    for sample in tqdm.tqdm(dataset, "Get annotations"):
        shapes = np.append(shapes, sample['annot'][:, 2:4] - sample['annot'][:, 0:2], axis=0)

    return shapes


def _metric(bbox_shapes, k, thres):  # compute metric
    r = torch.tensor(bbox_shapes[:, None] / k[None])
    x = torch.min(r, 1. / r).min(2)[0]  # ratio metric
    best = x.max(1)[0]  # best_x
    return (best > 1. / thres).float().mean()  #  best possible recall


def _optimizer(bbox_shapes, init_anchor_scales, threshold, iterations=5000):
    anchor_shapes = _get_anchors(init_anchor_scales)
    best_bpr = _metric(bbox_shapes, anchor_shapes, threshold)
    best_scales = init_anchor_scales
    print("Init %s : %f\n" % (best_scales, best_bpr.item()))

    its = tqdm.tqdm(range(iterations))
    for i in its:
        its.set_description("Best possible recall: %f" % best_bpr)

        random_scales = np.random.rand(3) * (1, 1.5, 2.5)
        random_scales.sort()
        anchor_shapes = _get_anchors(random_scales)
        bpr = _metric(bbox_shapes, anchor_shapes, threshold)

        if bpr > best_bpr:
            best_scales = random_scales
            best_bpr = bpr

    anchor_shapes = _get_anchors(best_scales)
    best_bpr = _metric(bbox_shapes, anchor_shapes, threshold)

    print("Best %s : %f" % (best_scales, best_bpr.item()))

    return best_bpr, best_scales


def optimize_anchors(dataset, _with_box_shapes=False, use_cached=True):
    cached_anchors = load_cached_anchors()

    if use_cached and cached_anchors is not None:
        print("# Use cached anchors")
        return cached_anchors

    init_scales = [0.3, 0.5, 0.7]
    threshold = 1.7

    bbox_shapes = _get_bbox_shapes(dataset)
    bpr, best_scales = _optimizer(bbox_shapes, init_scales, threshold, iterations=500)

    if use_cached:
        save_cached_anchors(list(best_scales), bpr.item())

    if _with_box_shapes:
        return best_scales, bbox_shapes
    else:
        return best_scales


cached_anchors_filename = '_cached_anchors.json'


def load_cached_anchors():
    if not os.path.exists(cached_anchors_filename):
        return None

    cached_anchors = json.load(open(cached_anchors_filename, 'r'))
    if not isinstance(cached_anchors, dict) or 'scales' not in cached_anchors.keys():
        return None

    cached_scales = cached_anchors['scales']

    if not isinstance(cached_scales, list):
        return None
    else:
        return cached_scales


def save_cached_anchors(scales, bpr):
    json.dump({
        'scales': scales, 'bpr': bpr}, open(cached_anchors_filename, 'w'))


if __name__ == '__main__':
    from efficientdet.train import FlightDetectionDataset

    argparser = argparse.ArgumentParser()
    argparser.add_argument("--data_path", type=str, required=True)
    argparser.add_argument("--img_size", type=int, default=512)
    opt = argparser.parse_args()


    def _load_dataset(data_path, img_size=None):
        PATHS = {
            "train": (os.path.join(data_path, "images", "train"),
                      os.path.join(data_path, "annotations", 'instances_train.json')),
        }

        img_folder, ann_file = PATHS['train']
        dataset = FlightDetectionDataset(img_folder, ann_file, opt.whole_img_size, transform=transforms.Compose(
            [PrepareBatch(), Normalizer(), ResizerSquare(img_size), FitIntoGrid(128)]), use_dummy_images=True,
                                         split_mode=SplitMode.SQUARE)

        return dataset


    dataset = _load_dataset(opt.data_path, opt.img_size)
    best_scales, bbox_shapes = optimize_anchors(dataset, _with_box_shapes=True, use_cached=False)
    anchor_shapes = _get_anchors(best_scales)

    plt.figure()
    plt.scatter(bbox_shapes[:, 0], bbox_shapes[:, 1])
    plt.scatter(anchor_shapes[:, 0], anchor_shapes[:, 1])
    plt.xlabel("Width")
    plt.ylabel("Height")
    plt.show()
