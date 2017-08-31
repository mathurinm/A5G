import numpy as np
import time
from scipy import sparse
from sklearn.datasets.mldata import fetch_mldata
from sklearn import preprocessing
from numpy.linalg import norm
from a5g import lasso_path
from a5g.utils import configure_plt, plot_res
from a5g.homotopy import blitz_path


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
                                       gap_spacing=1000, batch_size=10,
                                       max_updates=100000, max_iter=100,
                                       verbose=True, verbose_solver=False)
        dur_a5g_ns[t] = time.time() - t0

        t0 = time.time()
        betas_b, gaps_b = blitz_path(X, y, alphas, tol)
        dur_b[t] = time.time() - t0

    configure_plt()
    labels = ['A5G w. safe screening', 'A5G w/o safe screening', "Blitz"]
    plot_res([dur_a5g, dur_a5g_ns, dur_b], labels, tols, log=0)
