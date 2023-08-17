import random

from torchvision.transforms import transforms
from torchvision.transforms.functional import erase

from augment.utils import BaseAug, to_patch_size


class CutOut(BaseAug):
    def __init__(self, area_ratio=(0.02, 0.15), aspect_ratio=0.3, **kwargs):
        super().__init__(**kwargs, to_tensor=False)
        self.area_ratio = area_ratio
        self.aspect_ratio = aspect_ratio
        self.to_tensor = transforms.ToTensor()

    def __call__(self, img):
        img = self.to_tensor(img)
        h, w = img.shape[-2:]
        cut_w, cut_h = to_patch_size(w, h, self.area_ratio, self.aspect_ratio)

        aug_img = erase(
            img=img,
            i=int(random.uniform(0, h - cut_h)),
            j=int(random.uniform(0, w - cut_w)),
            h=cut_h,
            w=cut_w,
            v=0
        )
        return super().__call__(img, aug_img)


class CutPasteNormal(BaseAug):
    def __init__(self, area_ratio=(0.02, 0.15), aspect_ratio=0.3, **kwargs):
        super(CutPasteNormal, self).__init__(**kwargs)
        self.area_ratio = area_ratio
        self.aspect_ratio = aspect_ratio

    def __call__(self, img):
        h = img.size[0]
        w = img.size[1]
        cut_w, cut_h = to_patch_size(w, h, self.area_ratio, self.aspect_ratio)

        from_location_h = int(random.uniform(0, h - cut_h))
        from_location_w = int(random.uniform(0, w - cut_w))

        box = [from_location_w, from_location_h, from_location_w + cut_w, from_location_h + cut_h]
        patch = img.crop(box)

        if self.color_jitter:
            patch = self.color_jitter(patch)

        to_location_h = int(random.uniform(0, h - cut_h))
        to_location_w = int(random.uniform(0, w - cut_w))

        insert_box = [to_location_w, to_location_h, to_location_w + cut_w, to_location_h + cut_h]
        augmented = img.copy()
        augmented.paste(patch, insert_box)

        return super().__call__(img, augmented)


class CutPasteScar(BaseAug):
    """Randomly copy one patche from the image and paste it somewere else.
    Args:
        width (list): width to sample from. List of [min, max]
        height (list): height to sample from. List of [min, max]
        rotation (list): rotation to sample from. List of [min, max]
    """

    def __init__(self, width=(2, 16), height=(10, 25), rotation=(-45, 45),
                 **kwargs):
        super(CutPasteScar, self).__init__(**kwargs)
        self.width = width
        self.height = height
        self.rotation = rotation

    def __call__(self, img):
        h = img.size[0]
        w = img.size[1]

        # cut region
        cut_w = random.uniform(*self.width)
        cut_h = random.uniform(*self.height)

        from_location_h = int(random.uniform(0, h - cut_h))
        from_location_w = int(random.uniform(0, w - cut_w))

        box = [from_location_w, from_location_h, from_location_w + cut_w, from_location_h + cut_h]
        patch = img.crop(box)

        if self.color_jitter:
            patch = self.color_jitter(patch)

        # rotate
        rot_deg = random.uniform(*self.rotation)
        patch = patch.convert("RGBA").rotate(rot_deg, expand=True)

        # paste
        to_location_h = int(random.uniform(0, h - patch.size[0]))
        to_location_w = int(random.uniform(0, w - patch.size[1]))

        mask = patch.split()[-1]
        patch = patch.convert("RGB")

        augmented = img.copy()
        augmented.paste(patch, (to_location_w, to_location_h), mask=mask)

        return super().__call__(img, augmented)


class CutPasteUnion(object):
    def __init__(self, **kwargs):
        self.normal = CutPasteNormal(**kwargs)
        self.scar = CutPasteScar(**kwargs)

    def __call__(self, img):
        r = random.uniform(0, 1)
        if r < 0.5:
            return self.normal(img)
        else:
            return self.scar(img)


class CutPaste3Way(object):
    def __init__(self, **kwargs):
        self.normal = CutPasteNormal(**kwargs)
        self.scar = CutPasteScar(**kwargs)

    def __call__(self, img):
        org, cutpaste_normal = self.normal(img)
        _, cutpaste_scar = self.scar(img)
        return org, cutpaste_normal, cutpaste_scar
