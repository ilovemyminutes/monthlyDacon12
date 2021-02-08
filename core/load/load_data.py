import os
from glob import glob

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

from skimage import io, transform

import numpy as np
import pandas as pd


class dirtyMNISTDataset(Dataset):

    DATA_PATH = '../data/dirty_mnist/'
    
    def __init__(self, mode: str, transform: list=None):
        '''
        Args
        ---
        mode: str, 데이터셋 종류를 설정. 'train', 'valid', 'test'
        '''
        if mode in ['train', 'valid']:
            label_path = os.path.join(self.DATA_PATH, mode, mode+'_answer.csv')
            self.labels = pd.read_csv(label_path)
        self.img_path = os.path.join(self.DATA_PATH, mode)
        self.transform = transform
        return

    def __len__(self) -> int:
        """데이터셋의 크기를 return"""        
        return len(glob(os.path.join(self.img_path, '*.png')))

    def __getitem__(self, idx: int) -> dict:
        """이미지를 샘플링. 이미지와 레이블을 담은 딕셔너리 형태로 리턴

        Args:
            idx (int): index 값

        Returns:
            sample [type]: [description]
        """        
        if torch.is_tensor(idx): # 이게 뭘까
            idx = idx.tolist()

        filename = f'{self.labels.iloc[idx, 0]:0>5d}.png'
        img = io.imread(os.path.join(self.img_path, filename))

        label = np.array(self.labels.iloc[idx, 1:])
        sample = {'image': img, 'label': label}

        if self.transform:
            sample = self.transform(sample)
        return sample