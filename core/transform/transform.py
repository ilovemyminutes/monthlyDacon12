import torch
from torchvision import transforms


class TransformGenerator:
    def __init__(self, mode: str='baseline'):
        self.mode = mode
        return
    
    def generate(self, phase: str='train'):
        if self.mode == 'baseline':
            if phase == 'train':
                tr = transforms.Compose([
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomVerticalFlip(p=0.5),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        [0.485, 0.456, 0.406],
                        [0.229, 0.224, 0.225]
                    )
                ])
            else:
                tr = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(
                        [0.485, 0.456, 0.406],
                        [0.229, 0.224, 0.225]
                    )
                ])
        else:
            raise NotImplementedError()
        return tr