import numpy as np
import time
import matplotlib.pyplot as plt
import blitzl1
from scipy import sparse
from sklearn.datasets.mldata import fetch_mldata
from sklearn import preprocessing
from numpy.linalg import norm
from a5g import lasso_path
from a5g.utils import configure_plt


def plot_res(times_noscreen, times_screen_, times_b, tols):
    fig, ax = plt.subplots(figsize=(7, 3.7))
    # if I do color='r' in ax.bar() it's not the red I want so I use this hack:
    prop_list = list(plt.rcParams['axes.prop_cycle'])
    blue = prop_list[0]['color']
    green = prop_list[1]['color']
    red = prop_list[2]['color']
    width = 0.2
    ind = np.arange(len(tols))
    rects1 = ax.bar(ind - 1 * width, times_noscreen, width,
                    label='A5G w. safe screening',
                    color=blue)
    rects2 = ax.bar(ind + 0 * width, times_screen_, width,
                    label='A5G w/o safe screening',
                    color=green)
    rects3 = ax.bar(ind + 1 * width, times_b, width,
                    label='Blitz',
                    color=red)

    # add some text for labels, title and axes ticks
    ax.set_ylabel('Time (s)')
    # ax.set_title('Comparison of Blitz and A5G on Leukemia')
    ax.set_xticks(ind + width / 2)
    from matplotlib.ticker import FormatStrFormatter
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.3e'))
    ax.set_xticklabels(["%.0e" % tol for tol in tols])

    ax.set_xlabel(r"$\bar{\epsilon}$")
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()


def blitz_path(X, y, alphas, eps, max_iter=100):
    n_samples, n_features = X.shape
    n_alphas = alphas.shape[0]

    tol = eps * np.linalg.norm(y) ** 2
    blitzl1.set_tolerance(tol)
    blitzl1.set_use_intercept(0)
    prob = blitzl1.LassoProblem(X, y)

    blitzl1._num_iterations = max_iter
    betas = np.zeros((n_alphas, n_features))
    gaps = np.zeros(n_alphas)
    beta_init = np.zeros(n_features)

    for t in range(n_alphas):
        sol = prob.solve(alphas[t], initial_x=beta_init)
        beta_init = np.copy(sol.x)
        betas[t, :] = sol.x
        gaps[t] = sol.duality_gap

        if abs(gaps[t]) > tol:

            print("warning: did not converge, t = ", t)
            print("gap = ", gaps[t], "eps = ", eps)

    return betas, gaps


if __name__ == "__main__":
    # dataset_id = 'leukemia'
    dataset_id = 'finance'
    if dataset_id == "leukemia":
        leuk = fetch_mldata('leukemia')
        X = np.asfortranarray(leuk.data)
        y = leuk.target.astype(float)
        y -= np.mean(y)
        y /= np.linalg.norm(y)
    elif dataset_id == "finance":
        X = sparse.load_npz("./data/finance_filtered.npz")
        y = np.load("./data/finance_target.npy")

        preprocess = True
        if preprocess is True:
            X = preprocessing.normalize(X, axis=0)
            y -= np.mean(y)
            y /= norm(y)
        X.sort_indices()

    n_samples, n_features = X.shape
    alpha_max = np.max(np.abs(X.T.dot(y)))

    n_alphas = 4
    alphas = alpha_max * np.logspace(0, -.5, n_alphas)
    tols = np.logspace(-2, -8, 4)
    dur_a5g = np.zeros_like(tols)
    dur_a5g_ns = np.zeros_like(tols)
    dur_b = np.zeros_like(tols)
    for t, tol in enumerate(tols):
        t0 = time.time()
        betas, gaps = lasso_path(X, y, alphas, tol, screening=True,
                                 gap_spacing=1000,
                                 batch_size=10, max_updates=100000, max_iter=100,
                                 verbose=True,
                                 verbose_solver=False)
        dur_a5g[t] = time.time() - t0

        t0 = time.time()
        betas_ns, gaps_ns = lasso_path(X, y, alphas, tol, screening=False,
                                 gap_spacing=1000,
                                 batch_size=10, max_updates=100000, max_iter=100,
                                 verbose=True,
                                 verbose_solver=False)
        dur_a5g_ns[t] = time.time() - t0

        t0 = time.time()
        betas_b, gaps_b = blitz_path(X, y, alphas, tol)
        dur_b[t] = time.time() - t0

    configure_plt()
    plot_res(dur_a5g, dur_a5g_ns, dur_b, tols)


    # comparison of gaps and primal distances to objective.
