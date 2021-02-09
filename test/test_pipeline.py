import os
import unittest

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

# from main import get_data
from core.load.load_data import dirtyMNISTDataset
from core.model.cnn import VanillaCNN


class TestPipeline(unittest.TestCase):
    DATA_PATH = "data/dirty_mnist/"
    BATCH_SIZE = 8
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    EPOCHS = 1
    LEARNING_RATE = 1e-3
    CHANNEL_DIM = 1

    def test_dirtyMNISTDataset(self):
        train_dataset = dirtyMNISTDataset(
            mode="train", transform=None, data_path=self.DATA_PATH
        )
        valid_dataset = dirtyMNISTDataset(
            mode="valid", transform=None, data_path=self.DATA_PATH
        )

        self.assertEqual(True, isinstance(train_dataset, torch.utils.data.Dataset))
        self.assertEqual(True, isinstance(valid_dataset, torch.utils.data.Dataset))
        self.assertEqual(train_dataset.labels.shape[0], len(train_dataset))
        self.assertEqual(valid_dataset.labels.shape[0], len(valid_dataset))

    def test_VanillaCNN(self):
        train_dataset = dirtyMNISTDataset(
            mode="train", transform=None, data_path=self.DATA_PATH
        )
        train_dataloader = DataLoader(dataset=train_dataset, batch_size=self.BATCH_SIZE)
        sample = next(iter(train_dataloader))
        X, y = sample["image"], sample["label"].float().to(self.DEVICE)

        HEIGHT, WIDTH = X.size()[-2], X.size()[-1]

        model = VanillaCNN().to(self.DEVICE)
        loss_function = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=self.LEARNING_RATE)

        y_pred = model.forward(
            X.view(self.BATCH_SIZE, self.CHANNEL_DIM, HEIGHT, WIDTH)
            .float()
            .to(self.DEVICE)
        )
        loss = loss_function(y_pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        self.assertEqual(list(sample.keys()), ["image", "label"])
        self.assertEqual(type(sample["image"]), torch.Tensor)
        self.assertEqual(y_pred.size(), sample["label"].size())


if __name__ == "__main__":
    unittest.main()
