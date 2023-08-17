import argparse
import json
import os

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = '/data/jaeminy/data'


def get_objects(dataset):
    if dataset == 'mvtec':
        return [
            'bottle',
            'cable',
            'capsule',
            'carpet',
            'grid',
            'hazelnut',
            'leather',
            'metal_nut',
            'pill',
            'screw',
            'tile',
            'toothbrush',
            'transistor',
            'wood',
            'zipper',
        ]
    elif dataset == 'mpdd':
        return [
            'bracket_black',
            'bracket_brown',
            'bracket_white',
            'connector',
            'metal_plate',
            'tubes',
        ]
    else:
        raise ValueError(dataset)


def set_environment(seed=None, num_threads=16):
    torch.set_num_threads(num_threads)
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def save_json(out, path):
    with open(path, 'w') as f:
        json.dump(out, f, indent=4, sort_keys=True)


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False
