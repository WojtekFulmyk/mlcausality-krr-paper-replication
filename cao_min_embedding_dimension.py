import numpy as np
import numba


# Liangyue Cao's method for finding the minimum embedding dimension
# https://doi.org/10.1016/S0167-2789(97)00118-8
# Python implementation by Wojciech Fulmyk, 2023
# Returns d (a list of embedding dimensions), E1, and E2
@numba.jit(nopython=True)
def cao_min_embedding_dimension(
    x, maxd=10, tau=1, threshold=0.95, max_relative_change=0.1
):
    if maxd < 3:
        raise ValueError("maxd must be greater than or equal to 3")
    v = np.zeros(maxd)
    a = np.zeros(maxd)
    ya = np.zeros(maxd)
    E1 = np.zeros(maxd - 1)
    E2 = np.zeros(maxd - 1)
    ndp = x.flatten().shape[0]
    for d in range(1, maxd + 1):
        a[d - 1] = 0.0
        ya[d - 1] = 0.0
        ng = ndp - d * tau
        for i in range(ng):
            v0 = 1.0e30
            for j in range(ng):
                if j == i:
                    continue
                for k in range(d):
                    v[k] = abs(x[i + k * tau] - x[j + k * tau])
                if d != 1:
                    for k in range(d - 1):
                        v[k + 1] = max(v[k], v[k + 1])
                vij = v[d - 1]
                if vij < v0 and vij != 0.0:
                    n0 = j
                    v0 = vij
            a[d - 1] += max(v0, abs(x[i + d * tau] - x[n0 + d * tau])) / v0
            ya[d - 1] += abs(x[i + d * tau] - x[n0 + d * tau])
        a[d - 1] /= float(ng)
        ya[d - 1] /= float(ng)
        if d >= 2:
            E1[d - 2] = a[d - 1] / a[d - 2]
            E2[d - 2] = ya[d - 1] / ya[d - 2]
    d = np.array(list(range(1, maxd)))
    embedding_dim = np.nan
    for dimension in range(3, maxd + 1):
        relative_error = abs(E1[dimension - 2] - E1[dimension - 3]) / E1[dimension - 3]
        if (
            np.isnan(embedding_dim)
            and not np.isnan(E1[dimension - 2])
            and E1[dimension - 2] >= threshold
            and relative_error < max_relative_change
        ):
            embedding_dim = dimension - 2
            break
    return d, E1, E2, embedding_dim


# Confirm implementation works by reconstructing Fig 1
# from Cao's paper, which plots E1 and E2 for the
# Hénon attractor
# Note that some small discrepancies in Fig 1 of
# Cao's paper and the implementation here are normal
# because the points used here may not be the
# exact same ones used by Cao. Nonetheless,
# the conclusion is the same: the minimum
# embedding dimension is 2 and the figures
# are very similar, which confirms that the
# python implementation is valid.
def cao_plot_henon():
    import matplotlib.pyplot as plt

    def henon_map(xn, yn, a, b):
        return yn + 1 - (a * xn**2), b * xn

    xt = []
    # Initial x and y for henon_map
    x = 0
    y = 0
    for i in range(10000):
        xn, yn = henon_map(x, y, 1.4, 0.3)
        xt.append(x)
        x, y = xn, yn
    xt_10t = np.array(xt)
    xt_1t = xt_10t[-1000:]
    result_1t = cao_min_embedding_dimension(xt_1t, maxd=10, tau=1)
    result_10t = cao_min_embedding_dimension(xt_10t, maxd=10, tau=1)
    plt.plot(result_1t[0], result_1t[1], "D-", color="green", label="E1-1T")
    plt.plot(result_10t[0], result_10t[1], "+:", color="red", label="E1-10T")
    plt.plot(result_1t[0], result_1t[2], "s--", color="blue", label="E2-1T")
    plt.plot(result_10t[0], result_10t[2], "x-.", color="black", label="E2-10T")
    plt.legend(loc="lower left")
    print("result_1t embedding dimension: " + str(result_1t[3]))
    print("result_10t embedding dimension: " + str(result_10t[3]))
    plt.show(block=False)
    plt.pause(0.001)
