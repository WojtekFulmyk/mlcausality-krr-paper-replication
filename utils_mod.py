import numpy as np
from scipy import stats, io, signal, linalg
import matplotlib.pyplot as plt
from scipy.spatial.distance import squareform, pdist
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.svm import SVC
from sklearn import metrics
from scipy.stats import f as scipyf
import warnings
from os.path import join
import os
import math
from sklearn.cluster import KMeans
import torch
from normalize_0_mean_1_std import normalize_0_mean_1_std
from calc_f_stat import calc_f_stat
from multivariate_split import multivariate_split

"""
This code includes the main functionality of the proposed method. 
"""

"""
The file includes the primary function which calculates Granger causality using lsNGC. 
"""

def lsNGC(inp_series, ar_order=1, k_f=3, k_g=2, normalize=1):
    if normalize:
        X_normalized=normalize_0_mean_1_std(inp_series)
    else:
        X_normalized=inp_series.copy()
    
    X_train, Y_train , X_test, Y_test=multivariate_split(X=X_normalized,ar_order=ar_order)
    
    X_train=torch.flatten(X_train, start_dim=1)

    km= KMeans(n_clusters= k_f, max_iter= 100, random_state=123)
    km.fit(X_train)
    cent= km.cluster_centers_


    max=0 

    for i in range(k_f):
        for j in range(k_f):
            d= np.linalg.norm(cent[i]-cent[j])
            if(d> max):
                max= d
    d= max

    sigma= d/math.sqrt(2*k_f)

    sig_d=np.zeros((np.shape(X_normalized)[0],np.shape(X_normalized)[0]));
    sig=np.zeros((np.shape(X_normalized)[0],np.shape(X_normalized)[0]));

#    Z_train_label=Y_train
    for i in range(X_normalized.shape[0]):
        Z_temp=X_normalized.copy()
        Z_train, Z_train_label , _ , _=multivariate_split(X=Z_temp,ar_order=ar_order)
        Z_train=torch.flatten(Z_train, start_dim=1)
        Z_train_label=torch.flatten(Z_train_label, start_dim=1)

        # Obtain phase space Z_s by exclusing time series of of x_s
        Z_s_train, Z_s_train_label , _ , _=multivariate_split(X=np.delete(Z_temp,[i],axis=0),ar_order=ar_order)
        # Obtain phase space reconstruction of x_s
        W_s_train, W_s_train_label , _ , _=multivariate_split(X=np.array([Z_temp[i]]),ar_order=ar_order)

        # Flatten data
        Z_s_train=torch.flatten(Z_s_train, start_dim=1)
        Z_s_train_label=torch.flatten(Z_s_train_label, start_dim=1)

        W_s_train=torch.flatten(W_s_train, start_dim=1)
        W_s_train_label=torch.flatten(W_s_train_label, start_dim=1)
        # Obtain k_g number of cluster centers in the phase space W_s with k-means clustering, will have dim=(k_g * d)
        kmg= KMeans(n_clusters= k_g, max_iter= 100, random_state=123)
        kmg.fit(W_s_train)
        cent_W_s= kmg.cluster_centers_
        # Calculate activations for each of the k_g neurons
        shape= W_s_train.shape
        row= shape[0]
        column= k_g
        G= np.empty((row,column), dtype= float)
        maxg=0 

        for ii in range(k_g):
            for jj in range(k_g):
                dg= np.linalg.norm(cent_W_s[ii]-cent_W_s[jj])
                if(dg> maxg):
                    maxg= dg
        dg= maxg

        sigmag= dg/math.sqrt(2*k_g)
        if sigmag==0:
            sigmag=1
        for ii in range(row):
            for jj in range(column):
                dist= np.linalg.norm(W_s_train[ii]-cent_W_s[jj])
                G[ii][jj]= math.exp(-math.pow(dist,2)/math.pow(2*sigmag,2))
        # Generalized radial basis function
        g_ws=np.array([G[ii]/sum(G[ii]) for ii in range(len(G))])
        # Calculate activations for each of the k_f neurons 
        shape= Z_s_train.shape
        row= shape[0]
        column= k_f
        F= np.empty((row,column), dtype= float)
        for ii in range(row):
            for jj in range(column):
                cent_temp=cent.copy()
                cent_temp=np.delete(cent_temp,np.arange(jj,jj+ar_order),axis=1)
                dist= np.linalg.norm(Z_s_train[ii]-cent_temp)
                F[ii][jj]= math.exp(-math.pow(dist,2)/math.pow(2*sigma,2))
        # Generalized radial basis function
        f_zs=np.array([F[ii]/sum(F[ii]) for ii in range(len(F))])

        # Prediction in the presence of x_s
        num_samples=f_zs.shape[0]

        f_new=np.concatenate((0.5*f_zs,0.5*g_ws),axis=1)
        GTG= np.dot(f_new.T,f_new)
        GTG_inv= np.linalg.pinv(GTG)
        fac= np.dot(GTG_inv,f_new.T)
        W_presence= np.dot(fac,Z_train_label)
        
        prediction_presence= np.dot(f_new,W_presence)
        error_presence=prediction_presence-np.array(Z_train_label)
        sig[i,:]=np.diag(np.cov(error_presence.T))

        # Prediction without x_s
        GTG= np.dot(f_zs.T,f_zs)
        GTG_inv= np.linalg.pinv(GTG)
        fac= np.dot(GTG_inv,f_zs.T)
        W_absence= np.dot(fac,Z_train_label)

        prediction_absence= np.dot(f_zs,W_absence)
        error_absence=prediction_absence-np.array(Z_train_label)
        sig_d[i,:]=np.diag(np.cov(error_absence.T))
    # Comupte the Granger causality index

    Aff=np.log(np.divide(sig_d,sig))
    Aff=(Aff>0)*Aff
    np.fill_diagonal(Aff,0)
    f_stat=calc_f_stat(sig_d, sig, n=num_samples+1, pu=k_f+k_g, pr=k_f)
    np.fill_diagonal(f_stat,0)
    n=num_samples+1
    pu=k_f+k_g
    pr=k_f
    f_dfn = pu-pr
    f_dfd = n-pu-1
    pvalue = np.empty_like(f_stat)
    for i in range(f_stat.shape[0]):
        for j in range(f_stat.shape[1]):
            pvalue[i,j] = scipyf.sf(f_stat[i,j], f_dfn, f_dfd)
    
    return Aff, f_stat, pvalue
