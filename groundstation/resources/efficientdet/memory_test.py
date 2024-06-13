from train import EfficientDetModule
from data_loader.dataset_configuration import load_dataset_configuration
import torch
import tqdm
import time
import numpy as np
import utils.gpu_memory as gpu_memory


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

    model = EfficientDetModule(hparams).cuda()
    return model


@torch.no_grad()
def test(model, img_size=1024, runs=100):
    torch.cuda.empty_cache()
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


def train(model, img_size=1024, runs=100):
    torch.cuda.empty_cache()
    img = torch.rand((1, 3, img_size, img_size)).cuda()

    durations = []
    gpu_memory.tensor_mem_report()
    gpu_memory.print_cuda_allocated_memory()
    model = model.train()

    for _ in tqdm.tqdm(range(runs)):
        start = time.time()
        result = model(img, is_training=True)
        gpu_memory.tensor_mem_report()
        gpu_memory.print_cuda_allocated_memory()

        exit(-1)
        _ = result
        stop = time.time()
        durations.append(stop-start)

    median_duration = np.median(durations)

    return (1 / median_duration)

if __name__ == '__main__':
    for cv in [None, 'b4']:
        model = load_model(cv)
        fps = train(model)
        print(f"{cv} : {np.round(fps, 2)}")