from train import FasterRCNNModule
from data_loader.dataset_configuration import load_dataset_configuration
import torch
import tqdm
import time
import numpy as np


def load_model(config_variant):
    hparams = {}

    hparams['config_variant'] = config_variant
    hparams['type'] = 'whole'
    hparams['dataset'] = 'vis_drone'
    hparams['network'] = 'efficientdet'
    hparams['checkpoint_path'] = '/data/checkpoints'
    hparams['score_threshold'] = 0.01

    # load the dataset config to the configuration
    hparams = {**hparams, **load_dataset_configuration(hparams['network'],
                                                           hparams['dataset'],
                                                           hparams['type'],
                                                           hparams['config_variant'])}

    model = FasterRCNNModule(hparams).cuda()
    return model


@torch.no_grad()
def evaluate_fps(model, img_size=768, runs=100):
    img = torch.rand((1, 3, img_size, img_size)).cuda()

    durations = []
    model = model.eval()

    for _ in tqdm.tqdm(range(runs)):
        start = time.time()
        result = model(img, is_training=False)
        stop = time.time()
        durations.append(stop-start)

    median_duration = np.median(durations)

    return (1 / median_duration)


if __name__ == '__main__':
    for cv in [None, 'b4']:
        model = load_model(cv)
        fps = evaluate_fps(model)
        print(f"{cv} : {np.round(fps, 2)}")