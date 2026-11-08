"""PyTorch Dataset sınıfı."""
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .data_utils import augment, normalize, read_image, read_mask, resize_pair


class BrainMRIDataset(Dataset):
    def __init__(self, df: pd.DataFrame, img_size: int, mean, std, train: bool = False):
        self.df = df.reset_index(drop=True)
        self.img_size = img_size
        self.mean, self.std = mean, std
        self.train = train
        self.rng = np.random.default_rng()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = read_image(row["image_path"])
        mask = read_mask(row["mask_path"])
        img, mask = resize_pair(img, mask, self.img_size)
        if self.train:
            img, mask = augment(img, mask, self.rng)
        x = torch.from_numpy(normalize(img, self.mean, self.std))
        y = torch.from_numpy(mask[None].astype(np.float32))
        return x, y


def worker_init_fn(worker_id):
    """Her DataLoader işçisine farklı rastgele tohum verir.

    Aksi halde tüm işçiler aynı veri artırmayı üretebilir.
    """
    info = torch.utils.data.get_worker_info()
    seed = (torch.initial_seed() + worker_id) % (2**32)
    info.dataset.rng = np.random.default_rng(seed)
