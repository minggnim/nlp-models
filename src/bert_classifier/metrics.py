import numpy as np
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    confusion_matrix
)


def transform_outputs(outputs, targets):
    p = np.array(outputs).argmax(axis=-1)
    y = targets
    # y = np.array(targets).argmax(axis=1)  # for multilabel
    return y, p


def accuracy_metrics(outputs, targets):
    y, p = transform_outputs(outputs, targets)
    return accuracy_score(y, p)


def classification_metrics(outputs, targets):
    y, p = transform_outputs(outputs, targets)
    return dict([
        ('n', len(y)),
        ('accuracy', accuracy_score(y, p)),
        ('precision', precision_score(y, p, average='macro')),
        ('recall', recall_score(y, p, average='macro')),
        ('f1', f1_score(y, p, average='macro')),
        ('confusion', confusion_matrix(y, p))
    ])
