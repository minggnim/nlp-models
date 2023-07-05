'''
Performance metrics
'''
from typing import Literal
import numpy as np
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    confusion_matrix
)
import torch
from torchmetrics.classification import (
    MultilabelHammingDistance,
    MultilabelAccuracy
)


def transform_outputs(outputs, targets, multi_label=False):
    '''
    transform outputs to suitable format for calculating performance metrics
    '''
    preds = np.array(outputs).argmax(axis=-1)
    if multi_label:
        y = np.array(targets).argmax(axis=1)
    else:
        y = targets
    return y, preds


def accuracy_metrics(outputs, targets, multi_label=False):
    '''
    function to calculate accuracy
    outputs: model outputs
    targets: labels
    '''
    targets, preds = transform_outputs(outputs, targets, multi_label)
    return accuracy_score(targets, preds)


def classification_metrics(outputs, targets, multi_label=False):
    '''
    collection of classification metrics
    '''
    targets, preds = transform_outputs(outputs, targets, multi_label)
    return dict([
        ('n', len(targets)),
        ('accuracy', accuracy_score(targets, preds)),
        ('precision', precision_score(targets, preds, average='macro')),
        ('recall', recall_score(targets, preds, average='macro')),
        ('f1', f1_score(targets, preds, average='macro')),
        ('confusion', confusion_matrix(targets, preds))
    ])


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
