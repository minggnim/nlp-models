import torch


def cross_entropy_loss_fn(outputs, targets):
    return torch.nn.BCELoss()(outputs, targets)
