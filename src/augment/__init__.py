from . import old, ours
from .utils import collate_fn


def to_aug_function(name, use_ours=True, **kwargs):
    augment2 = ours.Identity2()

    if use_ours:
        if name == 'cutpaste':
            augment1 = ours.CutPaste(
                kwargs['area_ratio'],
                kwargs['aspect_ratio'],
                kwargs['angle']
            )
        elif name == 'cutout':
            augment1 = ours.CutOut(
                kwargs['area_ratio'],
                kwargs['aspect_ratio'],
                mode='black'
            )
        elif name == 'cutavg':
            augment1 = ours.CutOut(
                kwargs['area_ratio'],
                kwargs['aspect_ratio'],
                mode='average'
            )
        elif name == 'cutdiff':
            augment1 = ours.Identity1()
            augment2 = ours.SmoothCutOut(
                area_ratio=kwargs['area_ratio'],
                aspect_ratio=kwargs['aspect_ratio'],
            )
        else:
            raise ValueError()
    else:
        augment1 = {
            'normal': old.CutPasteNormal,
            'scar': old.CutPasteScar,
            '3way': old.CutPaste3Way,
            'union': old.CutPasteUnion,
            'cutout': old.CutOut,
        }[name]()

    return augment1, augment2
