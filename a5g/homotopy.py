import numpy as np
import scipy.sparse
from .lasso_fast import a5g, a5g_sparse


def lasso_path(X, y, alphas, tol,
               screening=True, gap_spacing=1000,
               batch_size=10, max_updates=50000, max_iter=100,
               verbose=False,
               verbose_solver=False):
    """Compute Lasso path with A5G as inner solver on dense or sparse X"""
    n_alphas = len(alphas)
    n_samples, n_features = X.shape

    sparse = scipy.sparse.issparse(X)
    if not sparse:
        if not np.isfortran(X):
            X = np.asfortranarray(X)
        solver = a5g
    else:
        if X.getformat() != 'csc' or not X.has_sorted_indices:
            raise ValueError("Give csc matrix with sorted indices")


    betas = np.zeros((n_alphas, n_features))
    final_gaps = np.zeros(n_alphas)

    # skip alpha_max and use decreasing alphas
    for t in range(1, n_alphas):
        if verbose:
            print("--------------------")
            print("Computing %dth alpha" % (t + 1))
        if t > 1:
            beta_init = betas[t - 1].copy()
            min_ws_size = max(len(np.where(beta_init != 0)[0]), 1)
        else:
            beta_init = betas[t]
            min_ws_size = 10

        alpha = alphas[t]
        if not sparse:
            sol = solver(X, y, alpha, beta_init,
                         max_iter, gap_spacing, max_updates, batch_size,
                         tol=tol, verbose=verbose_solver,
                         strategy=3, screening=screening,
                         min_ws_size=min_ws_size)
        else:
            sol = a5g_sparse(X.data, X.indices, X.indptr, y, alpha, beta_init,
                             max_iter, gap_spacing, max_updates, batch_size,
                             tol=tol, verbose=verbose_solver,
                             strategy=3, min_ws_size=min_ws_size,
                             screening=screening)
        betas[t], final_gaps[t] = sol[0], sol[2][-1]  # last gap
        if final_gaps[t] > tol:
            print("-------------WARNING: did not converge, t = %d" % t)
            print("gap = %.1e , tol = %.1e" % (final_gaps[t], tol))
    return betas, final_gaps
