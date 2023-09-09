# Get time samples and seed from arguments passed when this script is called
import sys

network_to_check = sys.argv[1]
n_time_samples_str = sys.argv[2]
n_time_samples_int = int(n_time_samples_str)
seed = int(sys.argv[3])

import os

# Set environment variables useful for multiprocessing.
# This will only work if your BLAS is MKL or openblas.
# If using other BLAS implementations, these have no effect
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

os.makedirs("plots", exist_ok=True)
os.makedirs("pickled_data", exist_ok=True)

import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=UserWarning)

import numpy as np

np.seterr(all="raise")

import pandas as pd

import cao_min_embedding_dimension
import mlcausality

from utils_mod import lsNGC as granger
from recovery_performance_mod import recovery_performance

from brier_loss import brier_loss_func

import causal_ccm

from tigramite import data_processing as pcmci_data_processing
from tigramite.pcmci import PCMCI
from tigramite.independence_tests.parcorr import ParCorr

import tqdm

import matplotlib.pyplot as plt

from copy import deepcopy
import itertools
from pprint import pprint
import pickle

import networkx as nx

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
)

# Get number of cpu cores
import psutil

num_cores = psutil.cpu_count(logical=True)
# Import multriprocessing plugins from tqdm
# These rely on concurrent.futures
from tqdm.contrib.concurrent import ensure_lock, process_map

# Set seed.
np.random.seed(seed)

if network_to_check == "5_linear":
    # from spectral_connectivity.simulate import simulate_MVAR
    def simulate_MVAR(
        coefficients,
        noise_covariance=None,
        n_time_samples=100,
        n_trials=1,
        n_burnin_samples=100,
    ):
        """
        Simulate multivariate autoregressive (MVAR) process.
        Parameters
        ----------
        coefficients : array, shape (n_time_samples, n_lags, n_signals, n_signals)
        noise_covariance : array, shape (n_signals, n_signals)
        Returns
        -------
        time_series : array, shape (n_time_samples - n_burnin_samples,
                                    n_trials, n_signals)
        """
        n_lags, n_signals, _ = coefficients.shape
        if noise_covariance is None:
            noise_covariance = np.eye(n_signals)
        time_series = np.random.multivariate_normal(
            np.zeros((n_signals,)),
            noise_covariance,
            size=(n_time_samples + n_burnin_samples, n_trials),
        )

        for time_ind in np.arange(n_lags, n_time_samples + n_burnin_samples):
            for lag_ind in np.arange(n_lags):
                time_series[time_ind, ...] += np.matmul(
                    coefficients[np.newaxis, np.newaxis, lag_ind, ...],
                    time_series[time_ind - (lag_ind + 1), ..., np.newaxis],
                ).squeeze()
        return time_series[n_burnin_samples:, ...]

    # Generate data using the spectral_connectivity package

    def baccala_example3():
        """Baccalá, L.A., and Sameshima, K. (2001). Partial directed coherence:
        a new concept in neural structure determination. Biological
        Cybernetics 84, 463–474.
        """
        np.random.seed(seed)
        n_time_samples, n_lags, n_signals = n_time_samples_int, 3, 5
        coefficients = np.zeros((n_lags, n_signals, n_signals))

        coefficients[0, 0, 0] = 0.95 * np.sqrt(2)
        coefficients[1, 0, 0] = -0.9025

        coefficients[1, 1, 0] = 0.50
        coefficients[2, 2, 0] = -0.40

        coefficients[1, 3, 0] = -0.5
        coefficients[0, 3, 3] = 0.5 * np.sqrt(2)
        coefficients[0, 3, 4] = 0.25 * np.sqrt(2)

        coefficients[0, 4, 3] = -0.5 * np.sqrt(2)
        coefficients[0, 4, 4] = 0.5 * np.sqrt(2)

        noise_covariance = None

        return (
            simulate_MVAR(
                coefficients,
                noise_covariance=noise_covariance,
                n_time_samples=n_time_samples,
                n_trials=50,
                n_burnin_samples=500,
            ),
            coefficients,
        )

    data, coefficients = baccala_example3()

    data_list = [data[:, i, :] for i in range(data.shape[1])]

    # Find the adjacency matrix (which are essentially the true labels)
    adjacency_matrix = np.zeros((coefficients.shape[1], coefficients.shape[2]))
    for i in range(coefficients.shape[0]):
        adjacency_matrix += np.ceil(np.abs(coefficients[i]))

    adjacency_matrix = np.clip(adjacency_matrix, 0, 1)
    np.fill_diagonal(adjacency_matrix, 0)
    adjacency_matrix = adjacency_matrix.T.astype(int)
    true_labels = adjacency_matrix
elif network_to_check == "5_nonlinear":
    # 5-node nonlinear network
    true_labels = np.array(
        [
            [0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 1, 0],
        ]
    )
    n_burnin_samples = 500
    n_trials = 50
    n_lags = 3
    data_list = []
    while True:
        np.random.seed(seed)
        seed += 1
        data = np.random.normal(0, 1, [n_lags, 5])
        try:
            for j in range(n_lags, n_time_samples_int + n_burnin_samples):
                x1 = (
                    0.95 * np.sqrt(2) * data[-1, 0]
                    - 0.9025 * data[-2, 0]
                    + np.random.normal(0, 1)
                )
                x2 = 0.5 * data[-2, 0] ** 2 + np.random.normal(0, 1)
                x3 = -0.4 * data[-3, 0] + np.random.normal(0, 1)
                x4 = (
                    -0.5 * data[-2, 0] ** 2
                    + 0.5 * np.sqrt(2) * data[-1, 3]
                    + 0.25 * np.sqrt(2) * data[-1, 4]
                    + np.random.normal(0, 1)
                )
                x5 = (
                    -0.5 * np.sqrt(2) * data[-1, 3]
                    + 0.5 * np.sqrt(2) * data[-1, 4]
                    + np.random.normal(0, 1)
                )
                data = np.vstack([data, np.array([x1, x2, x3, x4, x5])])
        except Exception:
            continue
        data = np.array(data[n_burnin_samples:])
        data_list.append(data)
        if len(data_list) == n_trials:
            break
elif network_to_check == "7_nonlinear":
    # 7-node nonlinear network
    true_labels = np.array(
        [
            [0, 1, 0, 0, 0, 1, 1],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 1],
            [0, 0, 0, 1, 0, 0, 0],
        ]
    )
    n_burnin_samples = 500
    n_trials = 50
    n_lags = 3
    data_list = []
    while True:
        np.random.seed(seed)
        seed += 1
        data = np.random.normal(0, 1, [n_lags, 7])
        try:
            for j in range(n_lags, n_time_samples_int + n_burnin_samples):
                x1 = (
                    0.95 * np.sqrt(2) * data[-1, 0]
                    - 0.9025 * data[-2, 0]
                    + np.random.normal(0, 1)
                )
                x2 = (
                    -0.04 * data[-3, 0] ** 3
                    + 0.04 * data[-1, 0] ** 3
                    + np.random.normal(0, 1)
                )
                x3 = (
                    -0.04 * np.sqrt(2) * data[-1, 1] ** 3
                    + 0.04 * np.sqrt(2) * data[-2, 1] ** 3
                    + np.random.normal(0, 1)
                )
                x4 = (
                    np.log1p(np.abs(data[-1, 2])) * np.sign(data[-1, 2])
                    + 0.001 * data[-2, 6] ** 3
                    - 0.001 * data[-3, 6] ** 3
                    + np.random.normal(0, 1)
                )
                x5 = np.clip(np.random.normal(0, 1), -1, 1) * 0.04 * data[
                    -2, 5
                ] ** 5 + np.random.normal(0, 1)
                x6 = (
                    0.04 * data[-2, 0] ** 3
                    + 0.04 * data[-1, 2] ** 3
                    + np.random.normal(0, 1)
                )
                x7 = np.clip(np.random.normal(0, 1), -0.5, 0.5) * (
                    0.04 * data[-2, 0] ** 3
                    + 0.1 * data[-1, 5] ** 2
                    - 0.1 * data[-2, 5] ** 2
                ) + np.random.normal(0, 1)
                data = np.vstack([data, np.array([x1, x2, x3, x4, x5, x6, x7])])
        except Exception:
            continue
        data = np.array(data[n_burnin_samples:])
        data_list.append(data)
        if len(data_list) == n_trials:
            break
elif network_to_check == "9_nonlinear":
    # 9-node nonlinear network
    true_labels = np.array(
        [
            [0, 1, 1, 1, 0, 0, 0, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
        ]
    )
    n_burnin_samples = 500
    n_trials = 50
    n_lags = 3
    data_list = []
    while True:
        np.random.seed(seed)
        seed += 1
        data = np.random.normal(0, 1, [n_lags, 9])
        try:
            for j in range(n_lags, n_time_samples_int + n_burnin_samples):
                x1 = (
                    0.95 * np.sqrt(2) * data[-1, 0]
                    - 0.9025 * data[-2, 0]
                    + np.random.normal(0, 1)
                )
                x2 = (
                    0.5 * data[-2, 0] ** 2
                    + 0.5 * data[-1, 1]
                    - 0.4 * data[-2, 1]
                    + np.random.normal(0, 1)
                )
                x3 = (
                    -0.4 * data[-3, 0]
                    + 0.5 * data[-1, 2]
                    - 0.4 * data[-2, 2]
                    + np.random.normal(0, 1)
                )
                x4 = (
                    -0.5 * data[-2, 0] ** 2
                    + 0.5 * data[-1, 3]
                    - 0.4 * data[-2, 3]
                    + 0.5 * np.sqrt(2) * data[-1, 3]
                    + 0.25 * np.sqrt(2) * data[-1, 4]
                    + np.random.normal(0, 1)
                )
                x5 = (
                    -0.5 * np.sqrt(2) * data[-1, 3]
                    + 0.5 * np.sqrt(2) * data[-1, 4]
                    + np.random.normal(0, 1)
                )
                x6 = (
                    np.log1p(np.abs(data[-1, 3])) * np.sign(data[-1, 3])
                    + 0.5 * data[-1, 5]
                    - 0.4 * data[-2, 5]
                    + np.random.normal(0, 1)
                )
                x7 = (
                    np.clip(np.random.normal(0, 1), -1, 1) * 0.04 * data[-2, 5] ** 5
                    + 0.5 * data[-1, 6]
                    - 0.4 * data[-2, 6]
                    + np.random.normal(0, 1)
                )
                x8 = (
                    0.4 * data[-2, 0]
                    + 0.25 * data[-1, 2] ** 3
                    + 0.5 * data[-1, 7]
                    - 0.4 * data[-2, 7]
                    + np.random.normal(0, 1)
                )
                x9 = (
                    np.clip(np.random.normal(0, 1), -0.5, 0.5)
                    * (
                        0.2 * data[-2, 0]
                        + 0.1 * data[-1, 7] ** 2
                        - 0.1 * data[-2, 7] ** 2
                    )
                    + 0.5 * data[-1, 8]
                    - 0.4 * data[-2, 8]
                    + np.random.normal(0, 1)
                )
                data = np.vstack([data, np.array([x1, x2, x3, x4, x5, x6, x7, x8, x9])])
        except Exception:
            continue
        data = np.array(data[n_burnin_samples:])
        data_list.append(data)
        if len(data_list) == n_trials:
            break
elif network_to_check == "11_nonlinear":
    # 11-node nonlinear network
    true_labels = np.array(
        [
            [0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0],
            [0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0],
            [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ]
    )
    n_burnin_samples = 500
    n_trials = 50
    n_lags = 3
    data_list = []
    while True:
        np.random.seed(seed)
        seed += 1
        data = np.random.normal(0, 1, [n_lags, 11])
        try:
            for j in range(n_lags, n_time_samples_int + n_burnin_samples):
                x1 = (
                    0.25 * data[-1, 0] ** 2
                    - 0.25 * data[-2, 0] ** 2
                    + np.random.normal(0, 1)
                )
                x2 = np.log1p(np.abs(data[-2, 0])) * np.sign(
                    data[-2, 0]
                ) + np.random.normal(0, 1)
                x3 = -0.1 * data[-3, 1] ** 3 + np.random.normal(0, 1)
                x4 = (
                    -0.5 * data[-2, 1] ** 2
                    + 0.5 * np.sqrt(2) * data[-1, 3]
                    + 0.25 * np.sqrt(2) * data[-1, 4]
                    + np.random.normal(0, 1)
                )
                x5 = (
                    -0.5 * np.sqrt(2) * data[-1, 3]
                    + 0.5 * np.sqrt(2) * data[-1, 4]
                    + np.random.normal(0, 1)
                )
                x6 = np.log1p(np.abs(data[-1, 3])) * np.sign(
                    data[-1, 3]
                ) + np.random.normal(0, 1)
                x7 = np.clip(np.random.normal(0, 1), -1, 1) * 0.04 * data[
                    -2, 5
                ] ** 5 + np.random.normal(0, 1)
                x8 = (
                    0.4 * data[-2, 0] + 0.25 * data[-1, 2] ** 3 + np.random.normal(0, 1)
                )
                x9 = np.clip(np.random.normal(0, 1), -0.5, 0.5) * (
                    0.2 * data[-2, 0] + 0.1 * data[-1, 7] ** 2 - 0.1 * data[-2, 7] ** 2
                ) + np.random.normal(0, 1)
                x10 = (
                    0.25 * data[-3, 0] ** 2
                    - 0.01 * data[-3, 1] ** 2
                    + 0.15 * data[-3, 2] ** 3
                    + np.random.normal(0, 1)
                )
                x11 = (
                    0.1 * data[-1, 1] ** 4
                    - 0.1 * data[-2, 1] ** 4
                    + 0.1 * data[-3, 5] ** 3
                    + np.random.normal(0, 1)
                )
                data = np.vstack(
                    [data, np.array([x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11])]
                )
        except Exception:
            continue
        data = np.array(data[n_burnin_samples:])
        data_list.append(data)
        if len(data_list) == n_trials:
            break
elif network_to_check == "34_zachary1":
    # Generate data
    n_burnin_samples = 500
    n_trials = 50
    n_lags = 1
    true_labels_list = []
    data_list = []
    while True:
        np.random.seed(seed)
        seed += 1
        # Load the Zachary karate club dataset
        G_orig = nx.karate_club_graph()
        adjacency_matrix = nx.adjacency_matrix(G_orig).todense()

        true_labels = np.clip(adjacency_matrix, 0, 1).astype(int)
        true_labels_list.append(true_labels)

        data = 0.01 * np.random.normal(0, 1, [n_lags, true_labels.shape[0]])
        try:
            for t in range(n_lags, n_time_samples_int + n_burnin_samples):
                data_new = np.zeros_like(data[[-1]])
                for i in range(data_new.shape[1]):
                    data_new[0, i] = (
                        (1 - 0.025 * true_labels[:, i].sum())
                        * (1 - 1.8 * data[-1, i] ** 2)
                        + np.matmul(
                            0.025 * true_labels[:, i], (1 - 1.8 * np.square(data[-1]))
                        )
                        + 0.01 * np.random.normal(0, 1)
                    )
                data = np.vstack([data, data_new])
        except Exception:
            continue
        data = np.array(data[n_burnin_samples:])
        data_list.append(data)
        if len(data_list) == n_trials:
            break
elif network_to_check == "34_zachary2":
    # Generate data
    n_burnin_samples = 500
    n_trials = 50
    n_lags = 1
    true_labels_list = []
    data_list = []
    while True:
        np.random.seed(seed)
        seed += 1
        # Load the Zachary karate club dataset
        G_orig = nx.karate_club_graph()
        adjacency_matrix = nx.adjacency_matrix(G_orig).todense()
        comb_list = deepcopy(
            list(itertools.combinations(range(adjacency_matrix.shape[0]), 2))
        )
        list_of_edges = [(i, j) for (i, j) in comb_list if adjacency_matrix[i, j] != 0]
        # Randomly select 5 edges to keep as bidirectional
        idx_bidirectional = np.random.choice(range(len(list_of_edges)), 5)
        edges_nonbidirectional = [
            list_of_edges[i]
            for i in range(len(list_of_edges))
            if i not in idx_bidirectional
        ]
        # Randomly assign a direction
        for a, b in edges_nonbidirectional:
            if np.random.random() < 0.5:
                adjacency_matrix[a, b] = 0
            else:
                adjacency_matrix[b, a] = 0

        true_labels = np.clip(adjacency_matrix, 0, 1).astype(int)
        true_labels_list.append(true_labels)

        data = 0.01 * np.random.normal(0, 1, [n_lags, true_labels.shape[0]])
        try:
            for t in range(n_lags, n_time_samples_int + n_burnin_samples):
                data_new = np.zeros_like(data[[-1]])
                for i in range(data_new.shape[1]):
                    data_new[0, i] = (
                        (1 - 0.05 * true_labels[:, i].sum())
                        * (1 - 1.8 * data[-1, i] ** 2)
                        + np.matmul(
                            0.05 * true_labels[:, i], (1 - 1.8 * np.square(data[-1]))
                        )
                        + 0.01 * np.random.normal(0, 1)
                    )
                data = np.vstack([data, data_new])
        except Exception as e:
            print(e)
            continue
        data = np.array(data[n_burnin_samples:])
        data_list.append(data)
        if len(data_list) == n_trials:
            break


if network_to_check != "34_zachary1" and network_to_check != "34_zachary2":
    true_labels_list = [true_labels for i in range(len(data_list))]


G = nx.from_numpy_array(true_labels_list[0], create_using=nx.DiGraph)
if network_to_check != "34_zachary1" and network_to_check != "34_zachary2":
    G_pos = nx.kamada_kawai_layout(G)
else:
    G_pos = nx.circular_layout(G)

G_d = dict(G.degree)
G_labels = {node: str(node + 1) for node in G_d.keys()}
nx.draw(
    G,
    G_pos,
    labels=G_labels,
    with_labels=True,
    node_color="white",
    node_size=750,
    edgecolors="black",
    linewidths=2,
    arrows=True,
    arrowstyle="->",
    connectionstyle="arc3, rad = 0.2",
)
plt.rcParams["svg.fonttype"] = "none"
if n_time_samples_str == "500":
    if network_to_check == "5_linear":
        plt.savefig("plots/5_linear_nonlinear_network.pdf", bbox_inches="tight")
        plt.savefig("plots/5_linear_nonlinear_network.eps", bbox_inches="tight")
    elif network_to_check == "5_nonlinear":
        pass
    else:
        plt.savefig("plots/" + network_to_check + "_network.pdf", bbox_inches="tight")
        plt.savefig("plots/" + network_to_check + "_network.eps", bbox_inches="tight")
# plt.show(block=False)
# plt.pause(0.001)


# Cao's minimum embedding dimension
pprint("### Find Cao's minimum embedding dimension ###")


def caoloop(d):
    d_list = [[] for i in range(d.shape[1])]
    E1_list = [[] for i in range(d.shape[1])]
    E2_list = [[] for i in range(d.shape[1])]
    embedding_dim_list = [[] for i in range(d.shape[1])]
    for c in range(d.shape[1]):
        (
            cur_d,
            cur_E1,
            cur_E2,
            embedding_dim,
        ) = cao_min_embedding_dimension.cao_min_embedding_dimension(
            d[:, c], maxd=20, tau=1, threshold=0.8, max_relative_change=0.3
        )
        d_list[c].append(cur_d)
        E1_list[c].append(cur_E1)
        E2_list[c].append(cur_E2)
        embedding_dim_list[c].append(embedding_dim)
    return [d_list, E1_list, E2_list, embedding_dim_list]


cao_loop_output = process_map(caoloop, data_list, max_workers=num_cores)
# Ensure process_map finished before continuing:
# the following will not pass until the lock can be acquired
with ensure_lock(tqdm.auto.tqdm, lock_name="mp_lock") as lk:
    pass

# Extract output into lists of lists for easier processing downstream
d_list = [[] for i in range(data_list[0].shape[1])]
E1_list = [[] for i in range(data_list[0].shape[1])]
E2_list = [[] for i in range(data_list[0].shape[1])]
embedding_dim_list = [[] for i in range(data_list[0].shape[1])]
for o in range(len(cao_loop_output)):
    for v in range(len(cao_loop_output[0][0])):
        d_list[v].append(cao_loop_output[o][0][v][0])
        E1_list[v].append(cao_loop_output[o][1][v][0])
        E2_list[v].append(cao_loop_output[o][2][v][0])
        embedding_dim_list[v].append(cao_loop_output[o][3][v][0])


min_embed_dim = round(np.nanmax(np.array(embedding_dim_list).flatten()))

pprint("Minimum embedding dimension: " + str(min_embed_dim))


# Function to calculate metrics
def calc_metrics(pvalue_matricies, thresh):
    # Check sensitivity and specificity
    sensitivities = []
    specificities = []
    accuracies = []
    balanced_accuracies = []
    f1wgt = []
    listlen = deepcopy(len(pvalue_matricies))
    for i in range(listlen):
        preds = deepcopy(np.array(pvalue_matricies[i])).astype(float)
        np.fill_diagonal(preds, np.nan)
        preds = preds.flatten()
        preds = preds[~np.isnan(preds)]
        preds = (preds < thresh).astype(int)
        true_labels_curr = deepcopy(true_labels_list[i]).astype(float)
        np.fill_diagonal(true_labels_curr, np.nan)
        true_labels_curr = true_labels_curr.flatten()
        true_labels_curr = true_labels_curr[~np.isnan(true_labels_curr)]
        true_labels_curr = true_labels_curr.astype(int)
        tn, fp, fn, tp = confusion_matrix(
            true_labels_curr.flatten(), preds.flatten()
        ).ravel()
        sensitivity = tp / (tp + fn)
        sensitivities.append(sensitivity)
        specificity = tn / (tn + fp)
        specificities.append(specificity)
        accuracy = accuracy_score(true_labels_curr.flatten(), preds.flatten())
        accuracies.append(accuracy)
        balanced_accuracy = balanced_accuracy_score(
            true_labels_curr.flatten(), preds.flatten()
        )
        balanced_accuracies.append(balanced_accuracy)
        f1 = f1_score(true_labels_curr.flatten(), preds.flatten(), average="weighted")
        f1wgt.append(f1)
    return sensitivities, specificities, accuracies, balanced_accuracies, f1wgt


# Run mlcausality:
pprint("### Run mlcausality ###")
# Note: use minimum embedding dimension - 1 for lag if no logdiff is used and
# dimension - 2 for lag if logdiff is used
lag = max(min_embed_dim - 1, 1)


def locoloop(d):
    mlcausality_output = mlcausality.multiloco_mlcausality(
        d, lags=[lag], scaler_init_1="quantile", return_pvalue_matrix_only=True
    )
    return mlcausality_output


all_mlcausality_output = process_map(locoloop, data_list, max_workers=num_cores)
# Ensure process_map finished before continuing:
# the following will not pass until the lock can be acquired
with ensure_lock(tqdm.auto.tqdm, lock_name="mp_lock") as lk:
    pass

mlcausality_pvalue_matricies = all_mlcausality_output
mlcausality_adjacency_matricies = [1 - i for i in mlcausality_pvalue_matricies]

mlcausality_recovery_performance = recovery_performance(
    mlcausality_adjacency_matricies, true_labels_list
)
pprint("mlcausality recovery performance:")
pprint("Mean: " + str(mlcausality_recovery_performance.mean()))
pprint(
    pd.DataFrame(mlcausality_recovery_performance).quantile([0, 0.25, 0.5, 0.75, 1]).T
)

mlcausality_brier_score_loss = brier_loss_func(
    mlcausality_pvalue_matricies, true_labels_list
)
pprint("mlcausality brier score:")
pprint("Mean: " + str(mlcausality_brier_score_loss.mean()))
pprint(pd.DataFrame(mlcausality_brier_score_loss).quantile([0, 0.25, 0.5, 0.75, 1]).T)

# Check sensitivity and specificity for the mlcausality model
mlc_sens, mlc_spec, mlc_acc, mlc_bal_acc, mlc_f1wgt = calc_metrics(
    mlcausality_pvalue_matricies, 0.05
)


pprint("mlcausality sensitivities:")
pprint(pd.DataFrame(mlc_sens).quantile([0, 0.25, 0.5, 0.75, 1]).T)
pprint("mlcausality specificities:")
pprint(pd.DataFrame(mlc_spec).quantile([0, 0.25, 0.5, 0.75, 1]).T)
pprint("mlcausality (sensitivities*specificities)**0.5:")
pprint(
    pd.DataFrame((np.array(mlc_sens) * np.array(mlc_spec)) ** 0.5)
    .quantile([0, 0.25, 0.5, 0.75, 1])
    .T
)
pprint("mlcausality accuracies:")
pprint(pd.DataFrame(mlc_acc).quantile([0, 0.25, 0.5, 0.75, 1]).T)
pprint("mlcausality balanced accuracies:")
pprint(pd.DataFrame(mlc_bal_acc).quantile([0, 0.25, 0.5, 0.75, 1]).T)
pprint("mlcausality f1 scores:")
pprint(pd.DataFrame(mlc_f1wgt).quantile([0, 0.25, 0.5, 0.75, 1]).T)

optimal_j = 0
best_score = 0
for j in [round(r * 0.01, 2) for r in range(1, 100)]:
    mlc_sens2, mlc_spec2, mlc_acc2, mlc_bal_acc2, mlc_f1wgt2 = calc_metrics(
        mlcausality_pvalue_matricies, j
    )
    best_score_new = np.median((np.array(mlc_sens2) * np.array(mlc_spec2)) ** 0.5)
    if best_score_new > best_score:
        best_score = deepcopy(best_score_new)
        optimal_j = deepcopy(j)


mlc_sens2, mlc_spec2, mlc_acc2, mlc_bal_acc2, mlc_f1wgt2 = calc_metrics(
    mlcausality_pvalue_matricies, optimal_j
)


pprint("Optimal cutoff: " + str(optimal_j))
pprint("Optimal mlcausality sensitivities:")
pprint(pd.DataFrame(mlc_sens2).quantile([0, 0.25, 0.5, 0.75, 1]).T)
pprint("Optimal mlcausality specificities:")
pprint(pd.DataFrame(mlc_spec2).quantile([0, 0.25, 0.5, 0.75, 1]).T)
pprint("Optimal mlcausality (sensitivities*specificities)**0.5:")
pprint(
    pd.DataFrame((np.array(mlc_sens2) * np.array(mlc_spec2)) ** 0.5)
    .quantile([0, 0.25, 0.5, 0.75, 1])
    .T
)
pprint("Optimal mlcausality accuracies:")
pprint(pd.DataFrame(mlc_acc2).quantile([0, 0.25, 0.5, 0.75, 1]).T)
pprint("Optimal mlcausality balanced accuracies:")
pprint(pd.DataFrame(mlc_bal_acc2).quantile([0, 0.25, 0.5, 0.75, 1]).T)
pprint("Optimal mlcausality f1 scores:")
pprint(pd.DataFrame(mlc_f1wgt2).quantile([0, 0.25, 0.5, 0.75, 1]).T)


# Run lsNGC
pprint("### Run lsNGC ###")

k_f = 25
k_g = 5
ar_order = max(min_embed_dim - 1, 1)
while k_f >= 2:
    try:
        _ = granger(data_list[0].T, k_f=k_f, k_g=k_g, ar_order=ar_order, normalize=1)
        break
    except Exception:
        k_f -= 1

pprint(
    "lsNGC parameters: k_f="
    + str(k_f)
    + ", k_g="
    + str(k_g)
    + ", ar_order="
    + str(ar_order)
)


def lsNGCloop(d):
    lsNGC_output = granger(d.T, k_f=k_f, k_g=k_g, ar_order=ar_order, normalize=1)
    return lsNGC_output


all_lsNGC_output = process_map(lsNGCloop, data_list, max_workers=num_cores)
# Ensure process_map finished before continuing:
# the following will not pass until the lock can be acquired
with ensure_lock(tqdm.auto.tqdm, lock_name="mp_lock") as lk:
    pass

lsgc_adjacency_matricies = []
lsgc_fstat_matricies = []
lsgc_pvalue_matricies = []
for out in all_lsNGC_output:
    lsgc_adjacency_matricies.append(out[0])
    lsgc_fstat_matricies.append(out[1])
    pvalues_lsgc = out[2]
    np.fill_diagonal(pvalues_lsgc, 1)
    # lsgc_adjacency_matricies.append(1 - pvalues_lsgc)
    lsgc_pvalue_matricies.append(pvalues_lsgc)

lsgc_recovery_performance = recovery_performance(lsgc_fstat_matricies, true_labels_list)
# lsgc_recovery_performance = recovery_performance(lsgc_adjacency_matricies,true_labels_list)
pprint("lsNGC recovery performance:")
pprint("Mean: " + str(lsgc_recovery_performance.mean()))
pprint(pd.DataFrame(lsgc_recovery_performance).quantile([0, 0.25, 0.5, 0.75, 1]).T)

lsgc_brier_score_loss = brier_loss_func(lsgc_pvalue_matricies, true_labels_list)
pprint("lsgc brier score:")
pprint("Mean: " + str(lsgc_brier_score_loss.mean()))
pprint(pd.DataFrame(lsgc_brier_score_loss).quantile([0, 0.25, 0.5, 0.75, 1]).T)

# Check sensitivity and specificity for the lsNGC model
lsgc_sens, lsgc_spec, lsgc_acc, lsgc_bal_acc, lsgc_f1wgt = calc_metrics(
    lsgc_pvalue_matricies, 0.05
)


pprint("lsgc sensitivities:")
pprint(pd.DataFrame(lsgc_sens).quantile([0, 0.25, 0.5, 0.75, 1]).T)
pprint("lsgc specificities:")
pprint(pd.DataFrame(lsgc_spec).quantile([0, 0.25, 0.5, 0.75, 1]).T)
pprint("lsgc (sensitivities*specificities)**0.5:")
pprint(
    pd.DataFrame((np.array(lsgc_sens) * np.array(lsgc_spec)) ** 0.5)
    .quantile([0, 0.25, 0.5, 0.75, 1])
    .T
)
pprint("lsgc accuracies:")
pprint(pd.DataFrame(lsgc_acc).quantile([0, 0.25, 0.5, 0.75, 1]).T)
pprint("lsgc balanced accuracies:")
pprint(pd.DataFrame(lsgc_bal_acc).quantile([0, 0.25, 0.5, 0.75, 1]).T)
pprint("lsgc f1 scores:")
pprint(pd.DataFrame(lsgc_f1wgt).quantile([0, 0.25, 0.5, 0.75, 1]).T)

optimal_j = 0
best_score = 0
for j in [round(r * 0.01, 2) for r in range(1, 100)]:
    lsgc_sens2, lsgc_spec2, lsgc_acc2, lsgc_bal_acc2, lsgc_f1wgt2 = calc_metrics(
        lsgc_pvalue_matricies, j
    )
    best_score_new = np.median((np.array(lsgc_sens2) * np.array(lsgc_spec2)) ** 0.5)
    if best_score_new > best_score:
        best_score = deepcopy(best_score_new)
        optimal_j = deepcopy(j)


lsgc_sens2, lsgc_spec2, lsgc_acc2, lsgc_bal_acc2, lsgc_f1wgt2 = calc_metrics(
    lsgc_pvalue_matricies, optimal_j
)


pprint("Optimal cutoff: " + str(optimal_j))
pprint("Optimal lsgc sensitivities:")
pprint(pd.DataFrame(lsgc_sens2).quantile([0, 0.25, 0.5, 0.75, 1]).T)
pprint("Optimal lsgc specificities:")
pprint(pd.DataFrame(lsgc_spec2).quantile([0, 0.25, 0.5, 0.75, 1]).T)
pprint("Optimal lsgc (sensitivities*specificities)**0.5:")
pprint(
    pd.DataFrame((np.array(lsgc_sens2) * np.array(lsgc_spec2)) ** 0.5)
    .quantile([0, 0.25, 0.5, 0.75, 1])
    .T
)
pprint("Optimal lsgc accuracies:")
pprint(pd.DataFrame(lsgc_acc2).quantile([0, 0.25, 0.5, 0.75, 1]).T)
pprint("Optimal lsgc balanced accuracies:")
pprint(pd.DataFrame(lsgc_bal_acc2).quantile([0, 0.25, 0.5, 0.75, 1]).T)
pprint("Optimal lsgc f1 scores:")
pprint(pd.DataFrame(lsgc_f1wgt2).quantile([0, 0.25, 0.5, 0.75, 1]).T)


# Run causal_ccm
pprint("### Run causal_ccm ###")


def causal_ccm_loop(d):
    causal_ccm_output = causal_ccm.loop_causal_cmm(d.astype(np.float64), tau=1, E=2)
    return causal_ccm_output


all_causal_ccm_output = process_map(causal_ccm_loop, data_list, max_workers=num_cores)
# Ensure process_map finished before continuing:
# the following will not pass until the lock can be acquired
with ensure_lock(tqdm.auto.tqdm, lock_name="mp_lock") as lk:
    pass


# Generate pvalue adjacency matricies
permute_lists = list(itertools.permutations(range(data_list[0].shape[1]), 2))
causal_ccm_pvalue_matricies = []
for mlout in all_causal_ccm_output:
    cur_pvalue_matrix = np.ones([data_list[0].shape[1], data_list[0].shape[1]])
    for X_idx, y_idx in permute_lists:
        cur_pvalue_matrix[X_idx, y_idx] = mlout.loc[
            ((mlout.X == X_idx) & (mlout.y == y_idx)), "pvalue"
        ].iloc[0]
    np.fill_diagonal(cur_pvalue_matrix, 1)
    causal_ccm_pvalue_matricies.append(cur_pvalue_matrix)


# Generate adjacency matricies and check recovery performance
permute_lists = list(itertools.permutations(range(data_list[0].shape[1]), 2))
causal_ccm_adjacency_matricies = []
for mlout in all_causal_ccm_output:
    cur_pvalue_adjacency_matrix = np.zeros(
        [data_list[0].shape[1], data_list[0].shape[1]]
    )
    for X_idx, y_idx in permute_lists:
        cur_pvalue_adjacency_matrix[X_idx, y_idx] = mlout.loc[
            ((mlout.X == X_idx) & (mlout.y == y_idx)), "score"
        ].iloc[0]
    causal_ccm_adjacency_matricies.append(cur_pvalue_adjacency_matrix)

causal_ccm_recovery_performance = recovery_performance(
    causal_ccm_adjacency_matricies, true_labels_list
)
pprint("causal_ccm recovery performance:")
pprint("Mean: " + str(causal_ccm_recovery_performance.mean()))
pprint(
    pd.DataFrame(causal_ccm_recovery_performance).quantile([0, 0.25, 0.5, 0.75, 1]).T
)

causal_ccm_brier_score_loss = brier_loss_func(
    causal_ccm_pvalue_matricies, true_labels_list
)
pprint("causal_ccm brier score:")
pprint("Mean: " + str(causal_ccm_brier_score_loss.mean()))
pprint(pd.DataFrame(causal_ccm_brier_score_loss).quantile([0, 0.25, 0.5, 0.75, 1]).T)

# Check sensitivity and specificity for the causal_ccm model
(
    causal_ccm_sens,
    causal_ccm_spec,
    causal_ccm_acc,
    causal_ccm_bal_acc,
    causal_ccm_f1wgt,
) = calc_metrics(causal_ccm_pvalue_matricies, 0.05)


pprint("causal_ccm sensitivities:")
pprint(pd.DataFrame(causal_ccm_sens).quantile([0, 0.25, 0.5, 0.75, 1]).T)
pprint("causal_ccm specificities:")
pprint(pd.DataFrame(causal_ccm_spec).quantile([0, 0.25, 0.5, 0.75, 1]).T)
pprint("causal_ccm (sensitivities*specificities)**0.5:")
pprint(
    pd.DataFrame((np.array(causal_ccm_sens) * np.array(causal_ccm_spec)) ** 0.5)
    .quantile([0, 0.25, 0.5, 0.75, 1])
    .T
)
pprint("causal_ccm accuracies:")
pprint(pd.DataFrame(causal_ccm_acc).quantile([0, 0.25, 0.5, 0.75, 1]).T)
pprint("causal_ccm balanced accuracies:")
pprint(pd.DataFrame(causal_ccm_bal_acc).quantile([0, 0.25, 0.5, 0.75, 1]).T)
pprint("causal_ccm f1 scores:")
pprint(pd.DataFrame(causal_ccm_f1wgt).quantile([0, 0.25, 0.5, 0.75, 1]).T)

optimal_j = 0
best_score = 0
for j in [round(r * 0.01, 2) for r in range(1, 100)]:
    (
        causal_ccm_sens2,
        causal_ccm_spec2,
        causal_ccm_acc2,
        causal_ccm_bal_acc2,
        causal_ccm_f1wgt2,
    ) = calc_metrics(causal_ccm_pvalue_matricies, j)
    best_score_new = np.median(
        (np.array(causal_ccm_sens2) * np.array(causal_ccm_spec2)) ** 0.5
    )
    if best_score_new > best_score:
        best_score = deepcopy(best_score_new)
        optimal_j = deepcopy(j)


(
    causal_ccm_sens2,
    causal_ccm_spec2,
    causal_ccm_acc2,
    causal_ccm_bal_acc2,
    causal_ccm_f1wgt2,
) = calc_metrics(causal_ccm_pvalue_matricies, optimal_j)


pprint("Optimal cutoff: " + str(optimal_j))
pprint("Optimal causal_ccm sensitivities:")
pprint(pd.DataFrame(causal_ccm_sens2).quantile([0, 0.25, 0.5, 0.75, 1]).T)
pprint("Optimal causal_ccm specificities:")
pprint(pd.DataFrame(causal_ccm_spec2).quantile([0, 0.25, 0.5, 0.75, 1]).T)
pprint("Optimal causal_ccm (sensitivities*specificities)**0.5:")
pprint(
    pd.DataFrame((np.array(causal_ccm_sens2) * np.array(causal_ccm_spec2)) ** 0.5)
    .quantile([0, 0.25, 0.5, 0.75, 1])
    .T
)
pprint("Optimal causal_ccm accuracies:")
pprint(pd.DataFrame(causal_ccm_acc2).quantile([0, 0.25, 0.5, 0.75, 1]).T)
pprint("Optimal causal_ccm balanced accuracies:")
pprint(pd.DataFrame(causal_ccm_bal_acc2).quantile([0, 0.25, 0.5, 0.75, 1]).T)
pprint("Optimal causal_ccm f1 scores:")
pprint(pd.DataFrame(causal_ccm_f1wgt2).quantile([0, 0.25, 0.5, 0.75, 1]).T)


# Run pcmci
pprint("### Run pcmci ###")


def pmciloop(d):
    dataframe = pcmci_data_processing.DataFrame(d)
    parcorr = ParCorr()
    pcmci = PCMCI(dataframe=dataframe, cond_ind_test=parcorr, verbosity=0)
    results = pcmci.run_pcmci(
        tau_min=1, tau_max=min_embed_dim - 1, pc_alpha=None, alpha_level=0.05
    )
    pcmci_pvalues_matrix = results["p_matrix"].min(axis=2)
    np.fill_diagonal(pcmci_pvalues_matrix, 1)
    return pcmci_pvalues_matrix


pcmci_pvalue_matricies = process_map(pmciloop, data_list, max_workers=num_cores)
# Ensure process_map finished before continuing:
# the following will not pass until the lock can be acquired
with ensure_lock(tqdm.auto.tqdm, lock_name="mp_lock") as lk:
    pass

pcmci_adjacency_matricies = []
for i in range(len(pcmci_pvalue_matricies)):
    pcmci_pvalues_matrix_oneminus = 1 - pcmci_pvalue_matricies[i]
    np.fill_diagonal(pcmci_pvalues_matrix_oneminus, 0)
    pcmci_adjacency_matricies.append(pcmci_pvalues_matrix_oneminus)

pcmci_recovery_performance = recovery_performance(
    pcmci_adjacency_matricies, true_labels_list
)
pprint("pcmci recovery performance:")
pprint("Mean: " + str(pcmci_recovery_performance.mean()))
pprint(pd.DataFrame(pcmci_recovery_performance).quantile([0, 0.25, 0.5, 0.75, 1]).T)

pcmci_brier_score_loss = brier_loss_func(pcmci_pvalue_matricies, true_labels_list)
pprint("pcmci brier score:")
pprint("Mean: " + str(pcmci_brier_score_loss.mean()))
pprint(pd.DataFrame(pcmci_brier_score_loss).quantile([0, 0.25, 0.5, 0.75, 1]).T)

# Check sensitivity and specificity for the pcmci model
pcmci_sens, pcmci_spec, pcmci_acc, pcmci_bal_acc, pcmci_f1wgt = calc_metrics(
    pcmci_pvalue_matricies, 0.05
)


pprint("pcmci sensitivities:")
pprint(pd.DataFrame(pcmci_sens).quantile([0, 0.25, 0.5, 0.75, 1]).T)
pprint("pcmci specificities:")
pprint(pd.DataFrame(pcmci_spec).quantile([0, 0.25, 0.5, 0.75, 1]).T)
pprint("pcmci (sensitivities*specificities)**0.5:")
pprint(
    pd.DataFrame((np.array(pcmci_sens) * np.array(pcmci_spec)) ** 0.5)
    .quantile([0, 0.25, 0.5, 0.75, 1])
    .T
)
pprint("pcmci accuracies:")
pprint(pd.DataFrame(pcmci_acc).quantile([0, 0.25, 0.5, 0.75, 1]).T)
pprint("pcmci balanced accuracies:")
pprint(pd.DataFrame(pcmci_bal_acc).quantile([0, 0.25, 0.5, 0.75, 1]).T)
pprint("pcmci f1 scores:")
pprint(pd.DataFrame(pcmci_f1wgt).quantile([0, 0.25, 0.5, 0.75, 1]).T)

optimal_j = 0
best_score = 0
for j in [round(r * 0.01, 2) for r in range(1, 100)]:
    pcmci_sens2, pcmci_spec2, pcmci_acc2, pcmci_bal_acc2, pcmci_f1wgt2 = calc_metrics(
        pcmci_pvalue_matricies, j
    )
    best_score_new = np.median((np.array(pcmci_sens2) * np.array(pcmci_spec2)) ** 0.5)
    if best_score_new > best_score:
        best_score = deepcopy(best_score_new)
        optimal_j = deepcopy(j)


pcmci_sens2, pcmci_spec2, pcmci_acc2, pcmci_bal_acc2, pcmci_f1wgt2 = calc_metrics(
    pcmci_pvalue_matricies, optimal_j
)


pprint("Optimal cutoff: " + str(optimal_j))
pprint("Optimal pcmci sensitivities:")
pprint(pd.DataFrame(pcmci_sens2).quantile([0, 0.25, 0.5, 0.75, 1]).T)
pprint("Optimal pcmci specificities:")
pprint(pd.DataFrame(pcmci_spec2).quantile([0, 0.25, 0.5, 0.75, 1]).T)
pprint("Optimal pcmci (sensitivities*specificities)**0.5:")
pprint(
    pd.DataFrame((np.array(pcmci_sens2) * np.array(pcmci_spec2)) ** 0.5)
    .quantile([0, 0.25, 0.5, 0.75, 1])
    .T
)
pprint("Optimal pcmci accuracies:")
pprint(pd.DataFrame(pcmci_acc2).quantile([0, 0.25, 0.5, 0.75, 1]).T)
pprint("Optimal pcmci balanced accuracies:")
pprint(pd.DataFrame(pcmci_bal_acc2).quantile([0, 0.25, 0.5, 0.75, 1]).T)
pprint("Optimal pcmci f1 scores:")
pprint(pd.DataFrame(pcmci_f1wgt2).quantile([0, 0.25, 0.5, 0.75, 1]).T)


df_all_recovery_performance = pd.DataFrame(
    np.array(
        [
            mlcausality_recovery_performance,
            lsgc_recovery_performance,
            causal_ccm_recovery_performance,
            pcmci_recovery_performance,
        ]
    ).T,
    columns=["MLC", "lsNGC", "LM", "PCMCI"],
)
with open(
    "pickled_data/" + network_to_check + "_auc_" + n_time_samples_str + ".pickle", "wb"
) as f:  # should be 'wb' rather than 'w'
    pickle.dump(df_all_recovery_performance, f)


df_all_brier_score_loss = pd.DataFrame(
    np.array(
        [
            mlcausality_brier_score_loss,
            lsgc_brier_score_loss,
            causal_ccm_brier_score_loss,
            pcmci_brier_score_loss,
        ]
    ).T,
    columns=["MLC", "lsNGC", "LM", "PCMCI"],
)
with open(
    "pickled_data/" + network_to_check + "_brier_" + n_time_samples_str + ".pickle",
    "wb",
) as f:  # should be 'wb' rather than 'w'
    pickle.dump(df_all_brier_score_loss, f)


df_all_sensitivities = pd.DataFrame(
    np.array(
        [
            mlc_sens,
            lsgc_sens,
            causal_ccm_sens,
            pcmci_sens,
        ]
    ).T,
    columns=["MLC", "lsNGC", "LM", "PCMCI"],
)
with open(
    "pickled_data/"
    + network_to_check
    + "_sensitivity_"
    + n_time_samples_str
    + ".pickle",
    "wb",
) as f:  # should be 'wb' rather than 'w'
    pickle.dump(df_all_sensitivities, f)


df_all_specificities = pd.DataFrame(
    np.array(
        [
            mlc_spec,
            lsgc_spec,
            causal_ccm_spec,
            pcmci_spec,
        ]
    ).T,
    columns=["MLC", "lsNGC", "LM", "PCMCI"],
)
with open(
    "pickled_data/"
    + network_to_check
    + "_specificity_"
    + n_time_samples_str
    + ".pickle",
    "wb",
) as f:  # should be 'wb' rather than 'w'
    pickle.dump(df_all_specificities, f)


df_all_gmean = pd.DataFrame(
    np.array(
        [
            list((np.array(mlc_sens) * np.array(mlc_spec)) ** (1 / 2)),
            list((np.array(lsgc_sens) * np.array(lsgc_spec)) ** (1 / 2)),
            list((np.array(causal_ccm_sens) * np.array(causal_ccm_spec)) ** (1 / 2)),
            list((np.array(pcmci_sens) * np.array(pcmci_spec)) ** (1 / 2)),
        ]
    ).T,
    columns=["MLC", "lsNGC", "LM", "PCMCI"],
)
with open(
    "pickled_data/" + network_to_check + "_gmean_" + n_time_samples_str + ".pickle",
    "wb",
) as f:  # should be 'wb' rather than 'w'
    pickle.dump(df_all_gmean, f)


df_all_accuracies = pd.DataFrame(
    np.array([mlc_acc, lsgc_acc, causal_ccm_acc, pcmci_acc]).T,
    columns=["MLC", "lsNGC", "LM", "PCMCI"],
)
with open(
    "pickled_data/" + network_to_check + "_accuracy_" + n_time_samples_str + ".pickle",
    "wb",
) as f:  # should be 'wb' rather than 'w'
    pickle.dump(df_all_accuracies, f)


df_all_balanced_accuracies = pd.DataFrame(
    np.array(
        [
            mlc_bal_acc,
            lsgc_bal_acc,
            causal_ccm_bal_acc,
            pcmci_bal_acc,
        ]
    ).T,
    columns=["MLC", "lsNGC", "LM", "PCMCI"],
)
with open(
    "pickled_data/"
    + network_to_check
    + "_balanced_accuracy_"
    + n_time_samples_str
    + ".pickle",
    "wb",
) as f:  # should be 'wb' rather than 'w'
    pickle.dump(df_all_balanced_accuracies, f)


df_all_f1wgt = pd.DataFrame(
    np.array([mlc_f1wgt, lsgc_f1wgt, causal_ccm_f1wgt, pcmci_f1wgt]).T,
    columns=["MLC", "lsNGC", "LM", "PCMCI"],
)
with open(
    "pickled_data/" + network_to_check + "_f1wgt_" + n_time_samples_str + ".pickle",
    "wb",
) as f:  # should be 'wb' rather than 'w'
    pickle.dump(df_all_f1wgt, f)


df_all_sensitivities2 = pd.DataFrame(
    np.array(
        [
            mlc_sens2,
            lsgc_sens2,
            causal_ccm_sens2,
            pcmci_sens2,
        ]
    ).T,
    columns=["MLC", "lsNGC", "LM", "PCMCI"],
)
with open(
    "pickled_data/"
    + network_to_check
    + "_sensitivity2_"
    + n_time_samples_str
    + ".pickle",
    "wb",
) as f:  # should be 'wb' rather than 'w'
    pickle.dump(df_all_sensitivities2, f)


df_all_specificities2 = pd.DataFrame(
    np.array(
        [
            mlc_spec2,
            lsgc_spec2,
            causal_ccm_spec2,
            pcmci_spec2,
        ]
    ).T,
    columns=["MLC", "lsNGC", "LM", "PCMCI"],
)
with open(
    "pickled_data/"
    + network_to_check
    + "_specificity2_"
    + n_time_samples_str
    + ".pickle",
    "wb",
) as f:  # should be 'wb' rather than 'w'
    pickle.dump(df_all_specificities2, f)


df_all_gmean2 = pd.DataFrame(
    np.array(
        [
            list((np.array(mlc_sens2) * np.array(mlc_spec2)) ** (1 / 2)),
            list((np.array(lsgc_sens2) * np.array(lsgc_spec2)) ** (1 / 2)),
            list((np.array(causal_ccm_sens2) * np.array(causal_ccm_spec2)) ** (1 / 2)),
            list((np.array(pcmci_sens2) * np.array(pcmci_spec2)) ** (1 / 2)),
        ]
    ).T,
    columns=["MLC", "lsNGC", "LM", "PCMCI"],
)
with open(
    "pickled_data/" + network_to_check + "_gmean2_" + n_time_samples_str + ".pickle",
    "wb",
) as f:  # should be 'wb' rather than 'w'
    pickle.dump(df_all_gmean2, f)


df_all_accuracies2 = pd.DataFrame(
    np.array([mlc_acc2, lsgc_acc2, causal_ccm_acc2, pcmci_acc2]).T,
    columns=["MLC", "lsNGC", "LM", "PCMCI"],
)
with open(
    "pickled_data/" + network_to_check + "_accuracy2_" + n_time_samples_str + ".pickle",
    "wb",
) as f:  # should be 'wb' rather than 'w'
    pickle.dump(df_all_accuracies2, f)


df_all_balanced_accuracies2 = pd.DataFrame(
    np.array(
        [
            mlc_bal_acc2,
            lsgc_bal_acc2,
            causal_ccm_bal_acc2,
            pcmci_bal_acc2,
        ]
    ).T,
    columns=["MLC", "lsNGC", "LM", "PCMCI"],
)
with open(
    "pickled_data/"
    + network_to_check
    + "_balanced_accuracy2_"
    + n_time_samples_str
    + ".pickle",
    "wb",
) as f:  # should be 'wb' rather than 'w'
    pickle.dump(df_all_balanced_accuracies2, f)


df_all_f1wgt2 = pd.DataFrame(
    np.array([mlc_f1wgt2, lsgc_f1wgt2, causal_ccm_f1wgt2, pcmci_f1wgt2]).T,
    columns=["MLC", "lsNGC", "LM", "PCMCI"],
)
with open(
    "pickled_data/" + network_to_check + "_f1wgt2_" + n_time_samples_str + ".pickle",
    "wb",
) as f:  # should be 'wb' rather than 'w'
    pickle.dump(df_all_f1wgt2, f)


pprint("Program finished running!")
