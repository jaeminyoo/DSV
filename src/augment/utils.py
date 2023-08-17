import math
import random

import torch
from torchvision.transforms import transforms


def sample_from_range(min_value, max_value, scale='linear'):
    if scale == 'linear':
        return random.uniform(min_value, max_value)
    elif scale == 'log':
        log_min = math.log(min_value)
        log_max = math.log(max_value)
        return math.exp(random.uniform(log_min, log_max))
    else:
        raise ValueError()


def to_patch_size(width, height, area_ratio, aspect_ratio):
    assert len(area_ratio) == 2 and len(aspect_ratio) == 2
    area = sample_from_range(*area_ratio, scale='log') * width * height
    aspect = sample_from_range(*aspect_ratio, scale='log')
    cut_w = int(round(math.sqrt(area * aspect)))
    cut_h = int(round(math.sqrt(area / aspect)))
    return cut_w, cut_h


def collate_fn(batch):
    # BaseAug returns a tuple of long tuples.
    # We convert them into a list (of length 2) of tuples.
    return [torch.stack(imgs) for imgs in list(zip(*batch))]


class BaseAug:
    def __init__(self, color_jitter=0.1, transform=None, to_tensor=True,
                 normalize=True):
        transform_list = []
        if transform is not None:
            transform_list.append(transform)
        if to_tensor:
            transform_list.append(transforms.ToTensor())
        if normalize:
            transform_list.append(transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ))
        self.transform = transforms.Compose(transform_list)

        if color_jitter is None:
            self.color_jitter = None
        else:
            self.color_jitter = transforms.ColorJitter(
                brightness=color_jitter,
                contrast=color_jitter,
                saturation=color_jitter,
                hue=color_jitter
            )

    def __call__(self, ori_img, aug_img):
        if self.transform is not None:
            ori_img = self.transform(ori_img)
            aug_img = self.transform(aug_img)
        return ori_img, aug_img
