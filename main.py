import argparse
from tqdm import tqdm

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from core.model.cnn import VanillaCNN
from core.evaluation.evaluation import evaluate
from core.load.load_data import dirtyMNISTDataset


def get_config():
    parser = argparse.ArgumentParser()
    return

def get_data(batch_size: int = 16):
    print(f'Getting Data...')
    
    # transform = transforms.Compose([transforms.Grayscale(), transforms.ToTensor()])
    train_dataset = dirtyMNISTDataset(mode='train', transform=None)
    valid_dataset = dirtyMNISTDataset(mode='valid', transform=None)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=batch_size)
    
    return train_loader, valid_loader

def get_model(lr: float = 0.01, device: str):
    model = VanillaCNN().to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    return model, loss_function, optimizer

def train_model(model: nn.Module, 
                loss_function,
                optimizer, 
                train_loader: DataLoader, valid_loader: DataLoader, 
                epochs:int=3, batch_size:int=16, device:str=None):
    
    model.train()
    print_every = 1
    
    # train phase
    for epoch in range(epochs):

        loss_val_sum = 0

        for batch in tqdm(train_loader):
            X = batch['image'].to(device)
            y_true = batch['label'].to(device)

            if len(X.size()) == 3:  # channel=1
                N, H, W = X.size()
                C = 1
            else:  # channel=3
                N, C, H, W = X.size()

            y_pred = model.forward(X.view(-1, C, H, W).to(device))
            loss = loss_function(y_pred, y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_val_sum += loss
        
        if ((epoch % print_every) == 0) or (epoch == (epochs - 1)):
            loss_val_avg = loss_val_sum / len(train_loader)
            accr_val = evaluate(model=model, data_iter=valid_loader, device=device)

if __name__ == '__main__':
    config = get_config()
    


            
