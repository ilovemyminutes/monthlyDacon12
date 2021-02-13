import os
from glob import glob

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

from skimage import io, transform

import numpy as np
import pandas as pd


class dirtyMNISTDataset(Dataset):

    DATA_PATH = "data/dirty_mnist/"

    def __init__(self, mode: str, transform: list = None, data_path: str = None):
        """
        Args
        ---
        mode: str, 데이터셋 종류를 설정. 'train', 'valid', 'test'
        """
        if data_path is None:
            data_path = self.DATA_PATH

        if mode in ["train", "valid"]:
            label_path = os.path.join(data_path, mode, mode + "_answer.csv")
            self.labels = pd.read_csv(label_path)
        self.img_path = os.path.join(data_path, mode)
        self.transform = transform
        return

    def __len__(self) -> int:
        """데이터셋의 크기를 return"""
        return len(glob(os.path.join(self.img_path, "*.png")))

    def __getitem__(self, idx: int) -> dict:
        """이미지를 샘플링. 이미지와 레이블을 담은 딕셔너리 형태로 리턴

        Args:
            idx (int): index 값

        Returns:
            sample [type]: [description]
        """
        if torch.is_tensor(idx):  # 이게 뭘까
            idx = idx.tolist()

        filename = f"{self.labels.iloc[idx, 0]:0>5d}.png"
        img = io.imread(os.path.join(self.img_path, filename))

        label = np.array(self.labels.iloc[idx, 1:])
        sample = {"image": img, "label": label}

        if self.transform:
            sample = self.transform(sample)
        return sample


# baseline
class MnistDataset(Dataset):
    def __init__(
        self,
        dir: os.PathLike,
        image_ids: os.PathLike,
        transforms: Sequence[Callable]
    ) -> None:
        self.dir = dir
        self.transforms = transforms

        self.labels = {}
        with open(image_ids, 'r') as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                self.labels[int(row[0])] = list(map(int, row[1:]))

        self.image_ids = list(self.labels.keys())

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, index: int) -> Tuple[Tensor]:
        image_id = self.image_ids[index]
        image = Image.open(
            os.path.join(
                self.dir, f'{str(image_id).zfill(5)}.png')).convert('RGB')
        target = np.array(self.labels.get(image_id)).astype(np.float32)

        if self.transforms is not None:
            image = self.transforms(image)

        return image, target