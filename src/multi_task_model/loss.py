import torch


def cross_entropy_loss_fn(outputs, targets, multi_label=False):
    if multi_label:
        return torch.nn.BCELoss()(outputs, targets)
    else:
        return torch.nn.CrossEntropyLoss()(outputs, targets)
