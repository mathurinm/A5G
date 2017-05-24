import numpy as np
from numpy.linalg import norm
import time
from .multitask_fast import gram_mt_fast
from .utils import (mt_primal, mt_dual,
                    mt_feature_prios, mt_compute_alpha)


def create_X_C(X, working_set):
    return X[:, working_set]


def mt_a5g(X, Y, alpha, max_iter=100, tol_ratio_inner=0.3,
           tol=1e-6, p0=100,
           verbose=False, screening=False, Beta_init=None,
           max_updates=10 ** 5, gap_spacing=1000, strategy=3,
           batch_size=10):
    """Used for M/EEG, for blocks equal to 1 line (fixed orient)"""
    assert np.isfortran(X)
    t0 = time.time()

    subpb_sizes = []
    subpb_n_iters = []
    n_samples, n_features = X.shape
    n_tasks = Y.shape[1]

    # for warm start use
    if Beta_init is not None:
        Beta = Beta_init.copy()
        R = Y - np.dot(X, Beta)
        Ksi = R / alpha
        Theta = np.zeros_like(Ksi)
    else:
        Ksi = Y / alpha
        Theta = np.zeros_like(Ksi)
        Beta = np.zeros((n_features, n_tasks))
        R = Y.copy()

    previously_active = np.array([], dtype=int)
    highest_d_obj = -np.inf
    ws_size = p0

    norms_X_col = norm(X, axis=0)
    norm_Y2 = (Y ** 2).sum()
    gaps = np.zeros(max_iter, dtype=float)
    times = np.zeros(max_iter, dtype=float)
    disabled = np.zeros(n_features, dtype=bool)

    XtY = np.dot(X.T, Y)

    for t in range(max_iter):

        scal = mt_compute_alpha(Theta, Ksi, X)
        if scal < 0:
            raise ValueError("negative alpha: %f" % scal)
        Theta = scal * Ksi + (1. - scal) * Theta

        d_obj = mt_dual(Y, Theta, alpha, norm_Y2)
        if d_obj > highest_d_obj:
            highest_d_obj = d_obj

        p_obj = mt_primal(R, Beta, alpha)
        gap = p_obj - highest_d_obj
        gaps[t] = gap
        times[t] = time.time() - t0
        if verbose:
            print("Iteration %d" % t)
            print("Primal {:.10f}".format(p_obj))
            print("Dual {:.10f}".format(highest_d_obj))
            print("Log 10 gap %f" % np.log10(gap))

        if np.isnan(gap):
            import pdb
            pdb.set_trace()
        if gap < tol:
            if gap < 0:
                print("!!!!!!!Negative gap: %e!!!!!!" % gap)
                break
            else:
                print("Gap smaller than %.2e" % tol,
                      "early exit after %d iterations" % t)
                break

        if screening:
            priorities = np.empty(n_features, dtype=float)
            priorities[disabled] = np.inf
            priorities[~disabled] = mt_feature_prios(Theta,
                                                     X[:, ~disabled],
                                                     norms_X_col[~disabled])
        else:
            priorities = mt_feature_prios(Theta, X, norms_X_col)
        priorities[previously_active] = - 1.

        if screening:
            radius = np.sqrt(2 * gap) / alpha
            disabled = priorities > radius
            if verbose:
                print("Subproblem %d, %d screened variables" %
                      (t, disabled.sum()))

        priorities_ix = np.argsort(priorities)
        C = priorities_ix[:ws_size]
        C.sort()

        subpb_sizes.append(len(C))
        if verbose:
            print("Solving subproblem with %d constraints" % len(C))
        tol_inner = tol_ratio_inner * gap

        X_C = create_X_C(X, C)

        gram = np.asfortranarray(np.dot(X_C.T, X_C))
        sol_subpb = gram_mt_fast(X_C, Y, alpha, Beta[C, :], gram, XtY[C, :],
                                 norm_Y2, tol_inner, max_updates, gap_spacing,
                                 strategy, batch_size)

        Beta_subpb, R, dual_scale, n_iter_ = sol_subpb[:4]
        Ksi = R / dual_scale  # = residuals /max(alpha, ||X^\top R||_\infty)

        subpb_n_iters.append(n_iter_)

        previously_active_subpb = np.where((Beta_subpb != 0).any(axis=1))[0]
        previously_active = C[previously_active_subpb]

        ws_size = 2 * len(previously_active)
        if ws_size < p0:
            ws_size = p0
        if ws_size > n_features:
            ws_size = n_features

        Beta = np.zeros((n_features, n_tasks))
        Beta[C, :] = Beta_subpb
    else:
        print("!!!!!!!! Outer solver did not converge !!!!!!!!")

    return Beta, gaps[:t + 1], times[:t + 1], subpb_sizes, subpb_n_iters
