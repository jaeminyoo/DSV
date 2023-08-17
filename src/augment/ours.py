import math
import random

import torch
from torch import nn
from torchvision.transforms import transforms

from augment.utils import to_patch_size


class Normalize:
    def __init__(self):
        super().__init__()
        self.transform = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

    def __call__(self, img):
        return self.transform(img)


class PostTransform:
    def __init__(self, to_tensor=True, normalize=True):
        transform = []
        if to_tensor:
            transform.append(transforms.ToTensor())
        if normalize:
            transform.append(Normalize())
        self.transform = transforms.Compose(transform)

    def __call__(self, img):
        return self.transform(img)


class Identity1:
    # Normalize only the original images.

    def __init__(self):
        super().__init__()
        self.transform1 = PostTransform(normalize=True)
        self.transform2 = PostTransform(normalize=False)

    def __call__(self, img):
        out = self.transform1(img)
        aug = self.transform2(img)
        return out, aug


class Identity2(nn.Module):
    def __init__(self):
        super().__init__()

    def __call__(self, img):
        return img


def _to_patch_aspect(aspect, aspect_range=(0.3, 3.3)):
    log_min = math.log(aspect_range[0])
    log_max = math.log(aspect_range[1])
    return torch.exp((log_max - log_min) * aspect + log_min)


def _to_patch_size(area, aspect, area_range=(0.02, 0.15), img_size=None):
    area_min = area_range[0]
    area_max = area_range[1]
    img_size = 1 if img_size is None else img_size
    area = (area_max - area_min) * area + area_min
    x = torch.sqrt(img_size * area * aspect)
    y = torch.sqrt(img_size * area / aspect)
    return x, y


def _to_grid(imgs):
    img_h, img_w = imgs.shape[-2:]
    mask1 = torch.arange(img_h, dtype=torch.float32, device=imgs.device) \
        .view(-1, 1).expand(-1, img_w)
    mask2 = torch.arange(img_w, dtype=torch.float32, device=imgs.device) \
        .view(1, -1).expand(img_h, -1)
    return torch.stack([mask1, mask2], dim=2)


class SmoothCutOut(nn.Module):
    def __init__(self, mode='gaussian', area_ratio=(0.02, 0.15), aspect_ratio=(0.3, 1.0)):
        super().__init__()
        self.mode = mode
        self.area_range = area_ratio
        self.aspect_range = aspect_ratio
        self.param = nn.Parameter(torch.tensor([0.5] * 2), requires_grad=True)
        self.normalize = Normalize()

    @staticmethod
    def _gau_patch(diff, scaler=0.1):
        out = torch.einsum('ijk,ijk->ij', diff, diff)
        return torch.exp(-out / (2 * scaler))

    @staticmethod
    def _box_patch(diff, temperature=0.1):
        out = torch.abs(2 * diff).max(dim=2)[0]
        return torch.sigmoid(-(out - 1) / temperature)

    def forward(self, imgs):
        device = imgs.device
        img_h, img_w = imgs.shape[-2:]
        aspect = _to_patch_aspect(self.param[1], self.aspect_range)
        patch_w, patch_h = _to_patch_size(self.param[0], aspect, self.area_range)

        pos_i = random.random()
        pos_j = random.random()
        mu = torch.tensor([img_h * pos_i, img_w * pos_j], device=device)
        sigma_inv = torch.stack([
            1 / (patch_h * img_h),
            1 / (patch_w * img_w),
        ]).to(device)

        diff = (_to_grid(imgs) - mu) * sigma_inv  # img_h x img_w x 2
        if self.mode == 'gaussian':
            out = self._gau_patch(diff)
        elif self.mode == 'box':
            out = self._box_patch(diff)
        else:
            raise ValueError(self.mode)
        return self.normalize(torch.clamp(imgs - out, 0, 1))


class CutOut:
    def __init__(self, area_ratio=(0.02, 0.15), aspect_ratio=(0.3, 1.0), mode='black'):
        super().__init__()
        self.area_ratio = area_ratio
        self.aspect_ratio = aspect_ratio
        self.to_tensor = transforms.ToTensor()
        self.transform = PostTransform(to_tensor=False)
        self.mode = mode

    def erase(self, img, i, j, h, w):
        if self.mode == 'black':
            v = 0
        elif self.mode == 'average':
            v = torch.mean(img[..., i:i + h, j:j + w])
        else:
            raise ValueError(self.mode)
        img = img.clone()
        img[..., i:i + h, j:j + w] = v
        return img

    def __call__(self, img):
        img = self.to_tensor(img)
        img_h, img_w = img.shape[-2:]
        cut_w, cut_h = to_patch_size(img_w, img_h, self.area_ratio, self.aspect_ratio)

        aug = self.erase(
            img=img,
            i=int(random.uniform(0, img_h - cut_h)),
            j=int(random.uniform(0, img_w - cut_w)),
            h=cut_h,
            w=cut_w,
        )
        img = self.transform(img)
        aug = self.transform(aug)
        return img, aug


class CutPaste:
    # Color jittering is not done on a patch.

    def __init__(self, area_ratio=(0.02, 0.15), aspect_ratio=(0.3, 1.0), angle=0):
        super().__init__()
        self.angle = angle
        self.area_ratio = area_ratio
        self.aspect_ratio = aspect_ratio
        self.transform = PostTransform()

    def __call__(self, img):
        img_h = img.size[0]
        img_w = img.size[1]
        cut_w, cut_h = to_patch_size(img_w, img_h, self.area_ratio, self.aspect_ratio)

        from_h = int(random.uniform(0, img_h - cut_h))
        from_w = int(random.uniform(0, img_w - cut_w))

        patch = img.crop((
            from_w,
            from_h,
            from_w + cut_w,
            from_h + cut_h,
        ))

        if self.angle != 0:
            patch = patch.convert('RGBA').rotate(self.angle, expand=True)

        to_h = int(random.uniform(0, img_h - patch.size[0]))
        to_w = int(random.uniform(0, img_w - patch.size[1]))

        aug = img.copy()
        if self.angle != 0:
            patch = patch.convert('RGB')
            aug.paste(patch, (to_w, to_h), mask=patch.split()[-1])
        else:
            aug.paste(patch, [to_w, to_h, to_w + cut_w, to_h + cut_h])

        img = self.transform(img)
        aug = self.transform(aug)
        return img, aug
