import numpy as np
from sklearn import metrics
from copy import deepcopy


def brier_loss_func(proba, label):
    proba = proba.copy()
    label = label.copy()
    N = len(proba)
    brier_all = np.zeros((N), dtype=np.float32)
    for i in range(N):
        label_matrix = deepcopy(label[i]).astype(np.float32)
        np.fill_diagonal(label_matrix, np.nan)
        label_matrix = label_matrix.flatten()
        label_matrix = label_matrix[~np.isnan(label_matrix)]
        label_matrix = label_matrix.astype(int)
        proba_matrix = deepcopy(proba[i])
        proba_matrix = np.round(proba_matrix, 16)
        proba_matrix = proba_matrix.astype(np.float32)
        np.fill_diagonal(proba_matrix, np.nan)
        proba_matrix = proba_matrix.flatten()
        proba_matrix = proba_matrix[~np.isnan(proba_matrix)]
        proba_matrix = np.clip(proba_matrix, a_min=0, a_max=1)
        brier_all[i] = metrics.brier_score_loss(label_matrix, proba_matrix, pos_label=0)
    return brier_all
