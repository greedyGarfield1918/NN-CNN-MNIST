import numpy as np


def accuracy(preds, labels):
    """
    Metric for MNIST.
    preds : [batch, D]
    labels : [batch, ]
    """
    assert preds.shape[0] == labels.shape[0]
    predict_label = np.argmax(preds, axis=-1)
    return (predict_label == labels).sum() / preds.shape[0]


def confusion_matrix(preds, labels, num_classes=10):
    """
    Compute confusion matrix.
    preds : [batch, D]  logits
    labels : [batch, ]
    Returns: [num_classes, num_classes]  matrix[i][j] = true label i predicted as j
    """
    predict_label = np.argmax(preds, axis=-1)
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for i in range(len(labels)):
        cm[labels[i], predict_label[i]] += 1
    return cm


def misclassified_indices(preds, labels):
    """
    Return indices of misclassified samples.
    preds : [batch, D]  logits
    labels : [batch, ]
    Returns: array of indices where prediction != label
    """
    predict_label = np.argmax(preds, axis=-1)
    return np.where(predict_label != labels)[0]
