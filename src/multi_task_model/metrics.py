from typing import Optional, Literal
import torch
from torchmetrics.classification import (
    MultilabelHammingDistance,
    MulticlassHammingDistance,
    MultilabelAccuracy,
    MulticlassAccuracy
)


def hamming_distance(
    outputs,
    targets,
    multi_label: bool = False, 
    num_labels: Optional[int] = None, 
    device: torch.device = torch.device('cpu'),
    average: Literal['micro', 'macro', 'weighted', 'none'] = 'macro'
):
    if not num_labels:
        raise ValueError('num_labels is required')
    if multi_label:
        hamming = MultilabelHammingDistance(num_labels, average=average)
    else:
        hamming = MulticlassHammingDistance(num_labels, average=average)
    hamming.to(device)
    return hamming(outputs, targets)


def accuracy(
    outputs,
    targets,
    num_labels: int,
    multi_label: bool = False,
    device: torch.device = torch.device('cpu'),
    average: Literal['micro', 'macro', 'weighted', 'none'] = 'macro'
):
    if not num_labels:
        raise ValueError('num_labels is required')
    if multi_label:
        acc = MultilabelAccuracy(num_labels, average=average)
    else:
        acc = MulticlassAccuracy(num_labels, average=average)
    acc.to(device)
    return acc(outputs, targets)
