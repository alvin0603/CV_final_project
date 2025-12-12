import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score


def compute_binary_metrics(all_scores, all_labels):
    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)

    auc = roc_auc_score(all_labels, all_scores)
    ap = average_precision_score(all_labels, all_scores)
    preds = (all_scores > 0.5).astype(int)
    acc = accuracy_score(all_labels, preds)

    max_conf_anom = all_scores[all_labels == 1].mean()
    max_conf_norm = all_scores[all_labels == 0].mean()

    return {
        "AUC": auc,
        "AP": ap,
        "ACC": acc,
        "MaxConf_Anom": max_conf_anom,
        "MaxConf_Norm": max_conf_norm
    }
