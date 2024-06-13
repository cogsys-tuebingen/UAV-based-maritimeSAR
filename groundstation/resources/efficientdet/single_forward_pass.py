from __future__ import division

from efficientdet.src.model import *

import os
import sys
import time
import datetime
import argparse
import tqdm
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

import torch
import numpy as np


def show_img(img):
    plt.figure()
    img = img.permute((1, 2, 0))
    plt.imshow(img.cpu().numpy())
    plt.show()


def test_split_for_img_size(splits, img_size, show_distribution=False, runs=1000):
    times = []
    times_splitted = []

    def predict(img):
        model.eval()
        with torch.no_grad():
            im0s = img.clone()
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Inference
            pred = model(img)[0]

            outputs = pred

        #model.train()

        return outputs

    for i in range(runs):
        # FIXME load the data
        img = torch.rand((1, 3, img_size, img_size), device=device)

        start_time = time.time()
        predict(img)
        needed_time = time.time() - start_time
        times.append(needed_time)

        start_time = time.time()
        assert img.shape[2] / splits % 1 == 0
        assert img.shape[3] / splits % 1 == 0

        splitted_img = torch.stack(torch.chunk(torch.stack(torch.chunk(img[0], splits, -2)), splits, -1)).reshape(
            -1, 3, int(img.shape[2] / splits), int(img.shape[3] / splits)
        )
        # splitted_img = img.reshape((split * split, 3, int(img.shape[2] / split), int(img.shape[3] / split)))
        # show_img(img[0])
        # show_img(splitted_img[1])
        predict(splitted_img)
        needed_time = time.time() - start_time
        times_splitted.append(needed_time)

    times = np.array(times)
    times_splitted = np.array(times_splitted)
    # print(times)
    avg_time = np.median(times)
    # print("Median inference time: %f => %.1f FPS" % (avg_time, 1 / avg_time))

    avg_time_splitted = np.median(times_splitted)
    # print("Median inference time (splitted): %f => %.1f FPS" % (avg_time_splitted, 1 / avg_time_splitted))

    if show_distribution:
        plt.figure()
        plt.hist(times, bins=20, label='Base')
        plt.hist(times_splitted, bins=20, label='Splitted')
        plt.legend()
        plt.show()

    return 1 / avg_time, 1 / avg_time_splitted


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    opt = parser.parse_args()
    print(opt)

    hparams = vars(opt)
    hparams['score_threshold'] = 0.01
    hparams['anchor_scales'] = [0.3, 0.5, 0.7]
    hparams['thres_found_object_iou'] = 0.25
    hparams['thres_not_found_object_iou'] = 0.15
    hparams['train_without_batchnorm'] = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load model
    criterion = FocalLoss(hparams['thres_found_object_iou'],
                               hparams['thres_not_found_object_iou'])
    anchors = Anchors(scales=hparams['anchor_scales'])
    model = EfficientDet(hparams, anchors, num_classes=80,
                         efficientnet_model='efficientnet-b2')
    model.to(device).eval()

    runs = 1000

    results = {}

    for img_size in tqdm.tqdm([512, 1024, 2048, 4096], "image size", position=0):
        for splits in tqdm.tqdm([2, 4, 8, 16], "splits", position=1):
            if img_size / splits < 216:
                # results[(img_size, splits)] = (1, 1)
                continue

            avg_time, avg_time_splitted = test_split_for_img_size(splits, img_size, runs=100,)
            results[(img_size, splits)] = (avg_time, avg_time_splitted)

    print(results)

    Xs = []
    Ys = []
    Zs = []
    Zs_fps = []
    for (img_size, splits) in results.keys():
        Xs.append(img_size)
        Ys.append(splits)
        fps, fps_splitted = results[(img_size, splits)]
        Zs.append(fps_splitted / fps)
        Zs_fps.append(fps)

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_trisurf(Xs, Ys, Zs, linewidth=0, antialiased=False, cmap=cm.coolwarm)
    plt.show()


    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_trisurf(Xs, Ys, Zs_fps, linewidth=0, antialiased=False, cmap=cm.coolwarm)
    plt.show()









