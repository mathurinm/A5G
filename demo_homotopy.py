import numpy as np
import time
from scipy import sparse
from sklearn.datasets.mldata import fetch_mldata
from sklearn import preprocessing
from numpy.linalg import norm
from a5g import lasso_path
from a5g.utils import configure_plt, plot_res, preprocess_data, primal
from a5g.homotopy import blitz_path


if __name__ == "__main__":
    n_alphas = 10

    # dataset_id = 'leukemia'
    dataset_id = 'finance'

    if dataset_id == "leukemia":
        leuk = fetch_mldata('leukemia')
        X = np.asfortranarray(leuk.data)
        y = leuk.target.astype(float)
        alpha_power_min = - 2
    elif dataset_id == "finance":
        X = sparse.load_npz("./data/finance_filtered.npz")
        y = np.load("./data/finance_target.npy")
        alpha_power_min = - 1.3

    preprocess = True
    if preprocess:
        X, y = preprocess_data(X, y)

    n_samples, n_features = X.shape
    alpha_max = np.max(np.abs(X.T.dot(y)))

    alphas = alpha_max * np.logspace(0, alpha_power_min, n_alphas)
    # tols = np.logspace(-2, -8, 4)
    tols = [1e-2]
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
                                       gap_spacing=1000, batch_size=10,
                                       max_updates=100000, max_iter=100,
                                       verbose=True, verbose_solver=False)
        dur_a5g_ns[t] = time.time() - t0

        t0 = time.time()
        betas_b, gaps_b = blitz_path(X, y, alphas, tol, use_intercept=1)
        dur_b[t] = time.time() - t0

    configure_plt()
    labels = ['A5G w. safe screening', 'A5G w/o safe screening', "Blitz"]
    plot_res([dur_a5g, dur_a5g_ns, dur_b], labels, tols, log=0,
             savepath=dataset_id + "_path.pdf")
