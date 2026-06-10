#!/usr/bin/env python

import torch
import numpy as np

def get_pred_distribution(masked_preds, num_classes):
    """
    Calculates distribution of class predictions.
    
    Input: 
        masked_preds : 1D torch.Tensor[int] - 1d tensor of predicted classes at positions where the truth label is the class of interest.
        num_classes : int - number of classes that can be possibly predicted.
    Output:
        dict[int:int]: distribution of predictions for the positions of interest, keys are classes, values are counts.
    """
    res = {k:0 for k in range(num_classes)}
    for i in masked_preds:
        res[i.item()] += 1
    return res

def calc_conf_mtx(actual_pos, pred_pos, actual_neg, pred_neg):
    """
    Calculate True Positive, True Negative, False Positive and False Negative.
    
    Input:
        t_truth : torch.Tensor[int] - Truth labels 
        t_preds : torch.Tensor[int] - Predicted labels
        label : int - class of interest
        
    Output:
        int - True Positive
        int - True Negative
        int - False Positive
        int - False Negative
    """
    actual_positive = (t_truth == label)
    pred_positive = (t_preds == label)

    tp = (actual_positive & pred_positive).sum().item()
    fn = (actual_positive & ~pred_positive).sum().item()
    fp = (~actual_positive & pred_positive).sum().item()
    tn = (~actual_positive & ~pred_positive).sum().item()

    return tp, tn, fp, fn

def calc_metrics(tp, tn, fp, fn):
    """
    Calculate precision, recall, F1 score and accuracy.
    
    Input:
        tp: int - True positive
        tn: int - True negative
        fp: int - False positive
        fn: int - False negative
    Output:
        float - precision
        float - recall
        float - F1 score
        float - accuracy
    """
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = (2 * precision * recall) / (precision + recall)
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    
    return precision, recall, f1_score, accuracy