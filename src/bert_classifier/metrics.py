'''
Performance metrics
'''
import numpy as np
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    confusion_matrix
)


def transform_outputs(outputs, targets):
    '''
    transform outputs to suitable format for calculating performance metrics
    '''
    preds = np.array(outputs).argmax(axis=-1)
    # y = np.array(targets).argmax(axis=1)  # for multilabel
    return targets, preds


def accuracy_metrics(outputs, targets):
    '''
    function to calculate accuracy
    outputs: model outputs
    targets: labels
    '''
    targets, preds = transform_outputs(outputs, targets)
    return accuracy_score(targets, preds)


def classification_metrics(outputs, targets):
    '''
    collection of classification metrics
    '''
    targets, preds = transform_outputs(outputs, targets)
    return dict([
        ('n', len(targets)),
        ('accuracy', accuracy_score(targets, preds)),
        ('precision', precision_score(targets, preds, average='macro')),
        ('recall', recall_score(targets, preds, average='macro')),
        ('f1', f1_score(targets, preds, average='macro')),
        ('confusion', confusion_matrix(targets, preds))
    ])
