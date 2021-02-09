import torch
from torch import nn
from torch.utils.data import DataLoader


def evaluate(model: nn.Module, data_iter: DataLoader, device):
    with torch.no_grad():
        n_total, n_correct = 0, 0
        model.eval()
        for batch in data_iter:
            X = batch['image'].to(device)
            y_true = batch['label'].to(device)

            if len(X.size()) == 3: # channel=1
                N, H, W = X.size()
                C = 1
            else: # channel=3
                N, C, H, W = X.size()
        
            model_pred = model.forward(X.view(-1, C, H, W).to(device)) # 여기서 애초에 해주네
            _, y_pred = torch.max(model_pred.data, 1) # 1: reduce할 차원

            n_correct += (y_pred == y_true).sum().item() # 원소가 하나인 텐서의 경우 item()을 통해 값을 추출할 수 있음
            n_total += N

        val_accr = (n_correct / n_total)
        model.train()
    return val_accr