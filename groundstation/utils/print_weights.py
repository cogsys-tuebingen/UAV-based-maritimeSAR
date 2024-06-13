import torch
import pytorch_lightning
import argparse

from efficientdet.train import EfficientDetModule


def load_checkpoint(path):
    model = EfficientDetModule.load_from_checkpoint(path)
    return model


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str)
    opt = parser.parse_args()

    model = load_checkpoint(opt.path)
    print(model)
    print()
    print()
    print()
    print(list(model.model.parameters()))
