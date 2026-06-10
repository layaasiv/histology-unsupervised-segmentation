#!/usr/bin/env python

import torch

def calc_conf_mtx(truth, preds, num_classes):
    """
    Generate the confusion matrix for 1 image

    Input:
        truth: torch.Tensor[int] - Matrix of predicted classes for each pixel.
        preds: torch.Tensor[int] - Matrix of true classes for each pixel.
        num_classes: int - Number of classes present in the image of interest.
    Output:
        torch.Tensor[int] (num_classes x num_classes) - Confusion matrix (rows = True label, column = Predicted label).
    """
    truth = truth.flatten().long()
    preds = preds.flatten().long()

    idx = truth * num_classes + preds # unique codes each (pred, truth) pair of labels

    conf = torch.bincount(
        idx, minlength=num_classes * num_classes
    ).reshape(num_classes, num_classes)

    return conf

def truefalse_posneg(conf):
    """
    Calculates true/false positives/negatives. 

    Input:
        conf: torch.Tensor[int] (num_classes x num_classes) - Confusion matrix.
    Output:
        torch.Tensor[int] (num_classes) x 4 - True/false positive/negative counts.
    """
    tp = conf.diag()
    fn = conf.sum(dim=1) - tp
    fp = conf.sum(dim=0) - tp
    tn = conf.sum() - tp - fp - fn
    return tp, fp, tn, fn

def cls_stats(conf):
    """
    Calculates classification statistics: precision, recall, F1 score, and accuracy.

    Input:
        conf: torch.Tensor[int] (num_classes x num_classes) - Confusion matrix.
    Output:
        torch.Tensor[float] (num_classes) x 4 - Precision, recall, F1 score, accuracy
    """
    tp, fp, tn, fn = truefalse_posneg(conf)

    precision = tp.float() / (tp+fp).clamp(min=1)
    recall = tp.float() / (tp+fn).clamp(min=1)
    f1 = (2 * precision * recall) / (precision + recall).clamp(min=1e-8)
    accuracy = (tp + tn).float() / (tp + tn + fp + fn)

    return precision, recall, f1, accuracy
