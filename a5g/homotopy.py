import numpy as np
import scipy.sparse
import blitzl1
import time
from sklearn.linear_model import MultiTaskLasso
from .lasso_fast import a5g_lasso, a5g_lasso_sparse
from .multitask_fast import a5g_mt


def lasso_path(X, y, alphas, tol,
               screening=True, gap_spacing=1000,
               batch_size=10, max_updates=50000, max_iter=100,
               verbose=False,
               verbose_solver=False, times_per_alpha=False):
    """Compute Lasso path with A5G as inner solver on dense or sparse X"""
    n_alphas = len(alphas)
    n_samples, n_features = X.shape

    sparse = scipy.sparse.issparse(X)
    if not sparse:
        if not np.isfortran(X):
            X = np.asfortranarray(X)
        solver = a5g_lasso
    else:
        if X.getformat() != 'csc' or not X.has_sorted_indices:
            raise ValueError("Give csc matrix with sorted indices")

    betas = np.zeros((n_alphas, n_features))
    final_gaps = np.zeros(n_alphas)
    all_times = np.zeros(n_alphas)

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
        t0 = time.time()
        if not sparse:
            sol = solver(X, y, alpha, beta_init,
                         max_iter, gap_spacing, max_updates, batch_size,
                         tol=tol, verbose=verbose_solver,
                         strategy=3, screening=screening,
                         min_ws_size=min_ws_size)
        else:
            sol = a5g_lasso_sparse(X.data, X.indices, X.indptr, y, alpha, beta_init,
                             max_iter, gap_spacing, max_updates, batch_size,
                             tol=tol, verbose=verbose_solver,
                             strategy=3, min_ws_size=min_ws_size,
                             screening=screening)

        all_times[t] = time.time() - t0
        betas[t], final_gaps[t] = sol[0], sol[2][-1]  # last gap
        if final_gaps[t] > tol:
            print("-------------WARNING: did not converge, t = %d" % t)
            print("gap = %.1e , tol = %.1e" % (final_gaps[t], tol))
    if times_per_alpha:
        return betas, final_gaps, all_times
    else:
        return betas, final_gaps


def lasso_path_mt(X, Y, alphas, tol,
                  screening=True, gap_spacing=1000,
                  batch_size=10, max_updates=50000, max_iter=100,
                  verbose=False,
                  verbose_solver=False):
    n_alphas = len(alphas)
    n_samples, n_features = X.shape
    n_tasks = Y.shape[1]

    Betas = np.zeros((n_alphas, n_features, n_tasks))
    final_gaps = np.zeros(n_alphas)

    # skip alpha_max and use decreasing alphas
    for t in range(1, n_alphas):
        if verbose:
            print("--------------------")
            print("Computing %dth alpha" % (t + 1))
        if t > 1:
            Beta_init = Betas[t - 1].copy()
            min_ws_size = np.sum(Beta_init.any(axis=0))
            min_ws_size = max(1, min_ws_size)
        else:
            Beta_init = Betas[t]
            min_ws_size = 10

        alpha = alphas[t]
        sol = a5g_mt(X, Y, alpha, Beta_init,
                     max_iter, gap_spacing, max_updates, batch_size,
                     tol=tol, verbose=verbose_solver,
                     strategy=3, screening=screening,
                     min_ws_size=min_ws_size)

        Betas[t], final_gaps[t] = sol[0], sol[2][-1]  # last gap
        if final_gaps[t] > tol:
            print("-------------WARNING: did not converge, t = %d" % t)
            print("gap = %.1e , tol = %.1e" % (final_gaps[t], tol))
    return Betas, final_gaps


def sklearn_path_mt(X, Y, alphas, tol):
    n_samples = X.shape[0]
    alphas_scaled = alphas / n_samples
    clf = MultiTaskLasso(alpha=None)
    return clf.path(X, Y, alphas=alphas_scaled, tol=tol, l1_ratio=1)


def blitz_path(X, y, alphas, eps, max_iter=1000, use_intercept=0,
               verbose=False, verbose_solver=0, times_per_alpha=False):
    n_samples, n_features = X.shape
    n_alphas = alphas.shape[0]

    tol = eps * np.linalg.norm(y) ** 2
    blitzl1.set_tolerance(tol)
    blitzl1.set_use_intercept(use_intercept)
    blitzl1.set_verbose(verbose_solver)
    prob = blitzl1.LassoProblem(X, y)

    blitzl1._num_iterations = max_iter
    betas = np.zeros((n_alphas, n_features))
    gaps = np.zeros(n_alphas)
    all_times = np.zeros(n_alphas)
    beta_init = np.zeros(n_features)

    for t in range(n_alphas):
        if verbose:
            print("--------------------")
            print("Computing %dth alpha" % (t + 1))
        t0 = time.time()
        sol = prob.solve(alphas[t], initial_x=beta_init)
        all_times[t] = time.time() - t0
        beta_init = np.copy(sol.x)
        betas[t, :] = sol.x
        gaps[t] = sol.duality_gap

        if abs(gaps[t]) > tol:
            print("warning: did not converge, t = ", t)
            print("gap = ", gaps[t], "eps = ", eps)

    if times_per_alpha:
        return betas, gaps, all_times
    else:
        return betas, gaps
