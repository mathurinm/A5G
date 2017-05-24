import numpy as np
from numpy.linalg import norm
import time
cimport numpy as np
from scipy.linalg.cython_blas cimport ddot, daxpy, dnrm2, dcopy, dger, dscal
from libc.math cimport fabs, sqrt, ceil
from libc.stdlib cimport rand, srand
cimport cython


cdef inline double fmax(double x, double y) nogil:
    return x if x > y else y


cdef inline int is_zero(int length, double[:] vector) nogil:
    cdef int i = 0
    for i in range(length):
        if vector[i] != 0:
            return 0

    return 1


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef double mt_primal_value(double alpha, int n_samples, int n_tasks,
                            int n_features, double[:, ::1] R,
                            double[:, ::1] W) nogil:
    cdef double p_obj = 0
    cdef int ii
    cdef int jj
    cdef int inc = 1
    for jj in range(n_features):
        # W is C ordered
        p_obj += dnrm2(&n_tasks, &W[jj, 0], &inc)
    p_obj *= alpha

    # compute fro norm of R line by line
    for ii in range(n_samples):
        # R is C ordered
        p_obj += dnrm2(&n_tasks, &R[ii, 0], &inc) ** 2 / 2
    return p_obj

# for UT
def mt_primal_c(double alpha, int n_samples, int n_tasks,
                int n_features, double[:, ::1] R, double[:, ::1] W):
    return mt_primal_value(alpha, n_samples, n_tasks, n_features, R, W)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef double mt_dual_value(double alpha, int n_samples, int n_tasks, double[:, ::1] R,
                   double[:, ::1] Y, double dual_scale, double norm_Y2) nogil:
    cdef double d_obj = 0
    cdef int ii
    cdef int jj
    for ii in range(n_samples):
        for jj in range(n_tasks):
            d_obj -= (Y[ii, jj] / alpha - R[ii, jj] / dual_scale) ** 2
    d_obj *= 0.5 * alpha ** 2
    d_obj += 0.5 * norm_Y2

    return d_obj

# for UT
def mt_dual_c(double alpha, int n_samples, int n_tasks, double[:, ::1] R,
                   double[:, ::1] Y, double dual_scale, double norm_Y2):
    return mt_dual_value(alpha, n_samples, n_tasks, R, Y, dual_scale, norm_Y2)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef double[:] block_ST(int n_tasks, double[:] w, double[:] grad_k, double alpha,
                       double norm_Xk2, double[:] w_new):
    """let u = w - grad_k
    This returns block_ST(u, alpha / norm_Xk2)), aka 0 if ||u|| <= alpha / norm_Xk2
    else (1 - alpha / (norm_Xk2 * ||u||)) * u
    """
    cdef int inc = 1
    cdef int t
    cdef double scaling
    # w_new = w_k - grad_k / norm_Xk2
    for t in range(n_tasks):
        w_new[t] = w[t] - grad_k[t] / norm_Xk2

    cdef double norm_tmp = dnrm2(&n_tasks, &w_new[0], &inc)

    scaling = fmax(0, 1. - alpha / (norm_Xk2 * norm_tmp))
    dscal(&n_tasks, &scaling, &w_new[0], &inc)

    return w_new

def block_ST_c(int n_tasks, double[:] w, double[:] grad_k, double alpha,
                       double norm_Xk2):
    w_new = np.zeros(n_tasks, dtype=np.float64)
    return block_ST(n_tasks, w, grad_k, alpha, norm_Xk2, w_new)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef double norm_difference(int n_tasks, double[:] w_k, double[:] w_k_new):
    cdef double nrm = 0.
    cdef int t
    for t in range(n_tasks):
        nrm += (w_k_new[t] - w_k[t]) ** 2
    return nrm


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def gram_mt_fast(double[::1, :] X,
                 double[:, ::1] Y,
                 double alpha,
                 double[:, ::1] W_init,
                 double[::1, :] Q,
                 double[:, ::1] XtY,
                 double norm_Y2,
                 double eps,
                 int max_updates,
                 int gap_spacing,
                 int strategy=3,
                 int batch_size=10,
                 int early_stop=1,
                 int monitor_time=0,
                 seed=0):

    cdef int n_samples = X.shape[0]
    cdef int n_features = X.shape[1]
    cdef int n_tasks = Y.shape[1]
    cdef int nb_batch = int(ceil(1.0 * n_features / batch_size))

    cdef double[:, ::1] W = W_init.copy()
    cdef int inc = 1
    cdef double minus_one = - 1.  # for cblas...
    cdef double one = 1  # ditto

    cdef double[:, ::1] gradients = np.dot(Q, W) - XtY
    cdef double XtR_axis1norm  # same as in sklearn: norm(XtR, axis=1)[j]
    cdef double dual_norm_XtR
    cdef double dual_scale

    cdef double[:, ::1] R = Y - np.dot(X, W)

    assert batch_size <= n_features
    assert strategy in (1, 2, 3)

    # cdef double[:] invnorm_Xcols_2 = 1. / np.diag(Q)
    # cdef double[:] alpha_invnorm_Xcols_2 = alpha / np.diag(Q)

    cdef double[:] tmp = np.zeros(n_tasks, dtype=np.float64)
    cdef double[:] w_ii = np.zeros(n_tasks, dtype=np.float64)

    cdef double gap
    cdef double[:] gaps = np.zeros(max_updates // gap_spacing, dtype=np.float64)
    cdef double d_obj
    cdef double highest_d_obj = 0.

    cdef int i
    cdef int ii
    cdef int jj
    cdef int k
    cdef int kk
    cdef int n_updates

    cdef double[:] update = np.zeros(n_tasks, dtype=np.float64)
    cdef double[:] w_k_new = np.zeros(n_tasks, dtype=np.float64)
    cdef double[:] w_ii_new = np.zeros(n_tasks, dtype=np.float64)
    cdef double norm_diff = 0.
    cdef double best_norm_diff = 0.
    cdef double t0 = time.time()
    cdef double t1
    cdef double flops = 0.

    srand(seed)
    # with nogil:
    if 1:
        for n_updates in range(max_updates):
            if (monitor_time == 0) and (n_updates % gap_spacing == 1):
                #  dual scale
                dual_norm_XtR = 0.
                for j in range(n_features):
                    XtR_axis1norm = dnrm2(&n_tasks, &gradients[j, 0], &inc)
                    if XtR_axis1norm > dual_norm_XtR:
                        dual_norm_XtR = XtR_axis1norm

                dual_scale = fmax(alpha, dual_norm_XtR)
                # R = Y - np.dot(X, W)
                for ii in range(n_samples):
                    for jj in range(n_tasks):
                        R[ii, jj] = Y[ii, jj] - ddot(&n_features,
                                                     &X[ii, 0], &n_samples,
                                                     &W[0, jj], &n_tasks)


                d_obj = mt_dual_value(alpha, n_samples, n_tasks, R, Y,
                                      dual_scale, norm_Y2)
                if d_obj > highest_d_obj:
                  highest_d_obj = d_obj

                gap = mt_primal_value(alpha, n_samples, n_tasks, n_features, R,
                                      W) - highest_d_obj
                gaps[n_updates // gap_spacing] = gap
                if early_stop and (gap < eps):
                    break

            elif (monitor_time == 1) and (n_updates % gap_spacing == 1):
                t1 = time.time()
                gaps[n_updates // gap_spacing] = t1 - t0 # store time not gap...

            # greedy:
            if strategy == 1:
                ii = 0
                best_norm_diff = 0.
                for j in range(batch_size):
                    # if batch_size == n_features, we perform full greedy
                    # otherwise we perform greedy among batch_size features
                    # randomly chose (this is NOT GS-rB)
                    if batch_size < n_features:
                        k = rand() % n_features
                    else:
                        k = j

                    w_k_new = block_ST(n_tasks, W[k, :], gradients[k, :],
                                      alpha, Q[k, k], w_k_new)
                    norm_diff = norm_difference(n_tasks, w_k_new, W[k, :])

                    if norm_diff >= best_norm_diff:
                        best_norm_diff = norm_diff
                        ii = k
                        # w_ii_new = w_k_new.copy():
                        dcopy(&n_tasks, &w_k_new[0], &inc, &w_ii_new[0], &inc)
            elif strategy == 2:  # cyclic
                ii = n_updates % n_features
                w_ii_new = block_ST(n_tasks, W[ii, :], gradients[ii, :],
                                   alpha, Q[ii, ii], w_ii_new)
            elif strategy == 3:
                # GS-rB
                start = (n_updates % nb_batch) * batch_size
                stop  = start + batch_size
                if stop > n_features:
                    stop = n_features
                ii = 0
                best_norm_diff = 0.
                for k in range(start, stop):
                    w_k_new = block_ST(n_tasks, W[k, :], gradients[k, :],
                                      alpha, Q[k, k], w_k_new)
                    norm_diff = norm_difference(n_tasks, w_k_new, W[k, :])

                    if norm_diff >= best_norm_diff:
                        best_norm_diff = norm_diff
                        ii = k
                        # w_ii_new = w_k_new.copy():
                        dcopy(&n_tasks, &w_k_new[0], &inc, &w_ii_new[0], &inc)

            # tmp = w_ii_new - W[ii, :] in two steps
            # tmp = w_ii_new
            dcopy(&n_tasks, &w_ii_new[0], &inc, &tmp[0], &inc)
            # tmp -= W[ii, :]
            daxpy(&n_tasks, &minus_one, &W[ii, 0], &inc, &tmp[0], &inc)

            if is_zero(n_tasks, tmp) == 0: # if tmp is not 0
                # update W[ii, :]
                dcopy(&n_tasks, &w_ii_new[0], &inc, &W[ii, 0], &inc)

                for kk in range(n_features):
                    for jj in range(n_tasks):
                        gradients[kk, jj] += Q[kk, ii] * tmp[jj]

        else:
           print('!!!! Inner solver did not converge !!!!')

    gaparray = np.array(gaps[:n_updates // gap_spacing + 1])
    return np.asarray(W), np.asarray(R), dual_scale, n_updates, gaparray
