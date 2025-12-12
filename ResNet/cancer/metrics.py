from sklearn.metrics import average_precision_score, roc_auc_score
import numpy as np
import torch

def compute_metrics(y_true, y_scores):
    """
    Computes Image AP and Image AUC.
    y_true: list or array of ground truth labels (0 or 1)
    y_scores: list or array of predicted probabilities/scores for class 1
    """
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)
    
    try:
        ap = average_precision_score(y_true, y_scores)
        auc = roc_auc_score(y_true, y_scores)
    except ValueError:
        ap = 0.0
        auc = 0.0
        
    return ap, auc
