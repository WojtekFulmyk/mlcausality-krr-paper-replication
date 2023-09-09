#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The function separates input time-series data into pieces of data, based on Taken's theorem to represent system dynamics within the state-space.

@Reference: 
Wismüller, A., Dsouza, A.M., Vosoughi, M.A. et al. 
Large-scale nonlinear Granger causality for inferring directed dependence from short multivariate time-series data. Sci Rep 11, 7817 (2021).
"""
import torch
import numpy as np


def multivariate_split(X,ar_order, valid_percent=0):
    X=X.copy()
    TS=np.shape(X)[1]
    n_vars=np.shape(X)[0]
    val_num=int(valid_percent*TS)
    my_data_train=torch.zeros((TS-ar_order-val_num,ar_order,n_vars))
    my_data_y_train=torch.zeros((TS-ar_order-val_num,1,n_vars))
    my_data_val=torch.zeros((val_num,ar_order,n_vars))
    my_data_y_val=torch.zeros((val_num,1,n_vars))
    for i in range(TS-ar_order-val_num):
        my_data_train[i]=torch.from_numpy(X.transpose()[i:i+ar_order,:])
        my_data_y_train[i]=torch.from_numpy(X.transpose()[i+ar_order,:])

    for i in range(TS-ar_order-val_num, TS-ar_order,1):
        my_data_val[i-(TS-ar_order-val_num)]=torch.from_numpy(X.transpose()[i:i+ar_order,:])
        my_data_y_val[i-(TS-ar_order-val_num)]=torch.from_numpy(X.transpose()[i+ar_order,:])
    return my_data_train, my_data_y_train, my_data_val, my_data_y_val
