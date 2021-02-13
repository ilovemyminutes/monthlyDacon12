import torch
from torch import nn
from torch.utils.data import DataLoader


def evaluate(model: nn.Module, data_iter: DataLoader, device):
    THRESHOLD = torch.autograd.Variable(torch.Tensor([.5])).to(device)
    with torch.no_grad():
        n_total, n_correct = 0, 0
        model.eval()
        for batch in data_iter:
            X = batch["image"].to(device)
            y_true = batch["label"].float().to(device)

            if len(X.size()) == 3:  # channel=1
                N, H, W = X.size()
                C = 1
            else:
                N, C, H, W = X.size()

            model_pred = model.forward(X.view(-1, C, H, W).float().to(device))
            y_pred = (model_pred > THRESHOLD).float().to(device) # binarize

            n_correct += (
                (y_pred == y_true).sum().item()
            )
            n_total += N

        val_accr = n_correct / n_total
        model.train()
    return val_accr
