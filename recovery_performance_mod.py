#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The code returns performance measures (in terms of AUROC) given a list of 
"Adjs": data matrices and 
"label": label matrices. 

@Reference: 

Originally created by:
Wismüller, A., Dsouza, A.M., Vosoughi, M.A. et al. 
Large-scale nonlinear Granger causality for inferring directed dependence from short multivariate time-series data. Sci Rep 11, 7817 (2021).

Modified by Wojciech (Victor) Fulmyk to account for the diagonal which is always labelled correctly.
"""
import numpy as np
from sklearn import metrics
from copy import deepcopy


def recovery_performance(Adjs,label):
    Adjs=Adjs.copy()
    label=label.copy()
    N=len(Adjs)
    auc_all = np.zeros((N), dtype=np.float32)   
    for i in range(N):
        label_matrix = deepcopy(label[i]).astype(float)
        np.fill_diagonal(label_matrix, np.nan)
        label_matrix = label_matrix.flatten()
        label_matrix = label_matrix[~np.isnan(label_matrix)]
        Adj_matrix = deepcopy(Adjs[i]).astype(float)
        np.fill_diagonal(Adj_matrix, np.nan)
        Adj_matrix = Adj_matrix.flatten()
        Adj_matrix = Adj_matrix[~np.isnan(Adj_matrix)]
        auc_all[i] = metrics.roc_auc_score(label_matrix, Adj_matrix)
    return auc_all

