from os.path import join
from pathlib import Path

from PIL import Image
from joblib import Parallel, delayed
from torch.utils.data import Dataset

import utils


class MVTecAD(Dataset):
    def __init__(self, root, obj_type, transform=None, mode='train', img_size=256):
        self.root = Path(root)
        self.transform = transform
        self.mode = mode
        self.img_size = img_size

        if self.mode == 'train':
            path = self.root / obj_type / 'train' / 'good'
            self.image_names = list(path.glob('*.png'))
        else:
            path = self.root / obj_type / 'test'
            self.image_names = list(path.glob(str(Path('*') / '*.png')))

        self.imgs = Parallel(n_jobs=10)(
            delayed(self.read_image)(file) for file in self.image_names
        )

    def read_image(self, file):
        return Image.open(file) \
            .resize((self.img_size, self.img_size)) \
            .convert('RGB')

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        if self.mode == 'train':
            img = self.imgs[idx].copy()
            if self.transform is not None:
                img = self.transform(img)
            return img
        else:
            file_name = self.image_names[idx]
            label = file_name.parts[-2]
            img = self.imgs[idx].copy()
            if self.transform is not None:
                img = self.transform(img)
            return img, label != 'good'


def load_data(dataset, obj_type, transform=None, mode='train', img_size=256):
    if dataset == 'mvtec':
        root = join(utils.DATA_PATH, 'mvtec_anomaly_detection')
        return MVTecAD(root, obj_type, transform, mode, img_size=img_size)
    elif dataset == 'mpdd':
        root = join(utils.DATA_PATH, 'MPDD')
        return MVTecAD(root, obj_type, transform, mode, img_size=img_size)
    else:
        raise ValueError()
