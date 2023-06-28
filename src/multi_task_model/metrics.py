from typing import Literal
import torch
from torchmetrics.classification import (
    MultilabelHammingDistance,
    MultilabelAccuracy
)


def hamming_distance(
    outputs,
    targets,
    num_labels: int,
    device: torch.device = torch.device('cpu'),
    average: Literal['micro', 'macro', 'weighted', 'none'] = 'macro'
):
    if not num_labels:
        raise ValueError('num_labels is required')
    hamming = MultilabelHammingDistance(num_labels, average=average)
    hamming.to(device)
    return hamming(outputs, targets)


def accuracy(
    outputs,
    targets,
    num_labels: int,
    device: torch.device = torch.device('cpu'),
    average: Literal['micro', 'macro', 'weighted', 'none'] = 'macro'
):
    if not num_labels:
        raise ValueError('num_labels is required')
    acc = MultilabelAccuracy(num_labels, average=average)
    acc.to(device)
    return acc(outputs, targets)