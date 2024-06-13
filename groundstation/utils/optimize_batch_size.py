import os
import json
import pytorch_lightning as lightning

cached_batch_size_filename = '_cached_batch_size.json'


def optimize_batch_size_for(model, use_cached=True):
    cached_batch_size = load_cached_batch_size()

    if use_cached and cached_batch_size is not None:
        print("# Use cached batch_size: %i" % cached_batch_size)
        return cached_batch_size

    print("# Optimize batch_size")

    trainer = lightning.Trainer(gpus=1, logger=False)
    new_batch_size = trainer.scale_batch_size(model, mode='binsearch')
    print("# Use batch_size=%i" % new_batch_size)

    if use_cached:
        save_cached_batch_size(new_batch_size)

    return new_batch_size


def load_cached_batch_size():
    if not os.path.exists(cached_batch_size_filename):
        return None

    cached_batch_size = json.load(open(cached_batch_size_filename, 'r'))
    if not isinstance(cached_batch_size, dict) or 'batch_size' not in cached_batch_size.keys():
        return None

    cached_scales = cached_batch_size['batch_size']

    if not isinstance(cached_scales, int):
        return None
    else:
        return cached_scales


def save_cached_batch_size(batch_size):
    json.dump({
        'batch_size': batch_size}, open(cached_batch_size_filename, 'w'))