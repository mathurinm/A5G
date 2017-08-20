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


@cython.boundscheck(False)
@cython.wraparound(False)
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
                            double[:, ::1] Beta) nogil:
    cdef double p_obj = 0
    cdef int ii
    cdef int jj
    cdef int inc = 1
    for jj in range(n_features):
        # W is C ordered
        p_obj += dnrm2(&n_tasks, &Beta[jj, 0], &inc)
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
                   double[:, ::1] Y, double dual_scale, double norm_Yfro) nogil:
    cdef double d_obj = 0
    cdef int ii
    cdef int jj
    for ii in range(n_samples):
        for jj in range(n_tasks):
            d_obj -= (Y[ii, jj] / alpha - R[ii, jj] / dual_scale) ** 2
    d_obj *= 0.5 * alpha ** 2
    d_obj += 0.5 * norm_Yfro

    return d_obj

# for UT
def mt_dual_c(double alpha, int n_samples, int n_tasks, double[:, ::1] R,
                   double[:, ::1] Y, double dual_scale, double norm_Yfro):
    return mt_dual_value(alpha, n_samples, n_tasks, R, Y, dual_scale, norm_Yfro)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void block_ST(int n_tasks, double * w, double * grad_k, double alpha,
                       double norm_Xk2, double * w_new):
    """let u = w - grad_k
    This computes block_ST(u, alpha / norm_Xk2)), aka
    0 if ||u|| <= alpha / norm_Xk2
    else (1 - alpha / (norm_Xk2 * ||u||)) * u
    and puts the result in w_new (last argument)
    """
    cdef int inc = 1
    cdef int t
    cdef double scaling
    # w_new = w_k - grad_k / norm_Xk2
    for t in range(n_tasks):
        w_new[t] = w[t] - grad_k[t] / norm_Xk2

    cdef double norm_tmp = dnrm2(&n_tasks, &w_new[0], &inc)

    scaling = fmax(0, 1. - alpha / (norm_Xk2 * norm_tmp))
    if scaling != 0:
        dscal(&n_tasks, &scaling, &w_new[0], &inc)
    else:
        for t in range(n_tasks):
            w_new[t] = 0.


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef double norm_difference(int n_tasks, double * w_k, double * w_k_new):
    cdef double nrm = 0.
    cdef int t
    for t in range(n_tasks):
        nrm += (w_k_new[t] - w_k[t]) ** 2
    return nrm


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def compute_gram(int n_samples, int ws_size, int[:] C, double[::1, :] X):
    cdef double[::1, :] gram = np.empty((ws_size, ws_size), order='F')
    cdef int inc = 1
    cdef int j
    cdef int k
    for j in range(ws_size):
        for k in range(j + 1):
            gram[k, j] = ddot(&n_samples, &X[0, C[k]], &inc,
                              &X[0, C[j]], &inc)
            gram[j, k] = gram[k, j]
    return gram


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void mt_set_feature_prios(int n_samples, int n_features,
                               int n_tasks, double[:, ::1] Theta,
                               double[::1, :] X, double * norms_X_col,
                               double * prios, int * disabled, double radius):
    cdef int j  # features
    cdef int k  # tasks
    cdef int inc = 1
    cdef double tmp
    cdef double[:] Xj_Theta = np.empty(n_tasks)
    for j in range(n_features):
        if disabled[j] == 1:
            prios[j] = radius + j
        else:
            for k in range(n_tasks):
                # Theta is not fortran
                Xj_Theta[k] = ddot(&n_samples, &X[0, j], &inc, &Theta[0, j], &n_tasks)
            tmp = dnrm2(&n_tasks, &Xj_Theta[0], &inc)
        prios[j] = (1 - tmp) / norms_X_col[j]


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef double mt_compute_alpha(double[:, ::1] Theta, double[:, ::1] Ksi,
                             double[::1, :] X):
    # TODO implement
    return 1.


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def mt_a5g(double[::1, :] X,
           double[:, ::1] Y,
           double alpha,
           double[:, ::1] Beta_init,
           int max_iter=50,
           double tol_ratio_inner=0.3,
           double tol=1e-6,
           int min_ws_size=10,
           int verbose=1,
           int screening=0,
           int max_updates=5 * 10**4,
           int gap_spacing=1000,
           int strategy=3,
           int batch_size=10
           ):
    # assert fortran
    cdef double t0 = time.time()
    cdef int n_samples = X.shape[0]
    cdef int n_features = X.shape[1]
    cdef int n_tasks = Y.shape[1]
    cdef int inc = 1
    cdef int i  # samples
    cdef int j  # features
    cdef int k  # tasks
    cdef int t  # outer loop iterations
    cdef int ws_size
    cdef double p_obj
    cdef double d_obj
    cdef double gap
    cdef double radius
    cdef double[:] prios = np.empty(n_features)
    cdef double[:, ::1] XtY = np.dot(X.T, Y)
    cdef double scal
    cdef double tmp
    cdef int tmpint

    cdef double[:] gaps = np.zeros(max_iter)
    cdef double[:] times = np.zeros(max_iter)
    cdef int[:] disabled = np.zeros(n_features, dtype=bool)
    cdef int n_disabled = 0

    cdef double[:, :: 1] Beta = np.empty([n_features, n_tasks])
    for j in range(n_features):
        for k in range(n_tasks):
            Beta[j, k] = Beta_init[j, k]
    R = Y - np.dot(X, Beta)
    cdef double[:, ::1] Ksi = R / alpha
    cdef double[:, ::1] Theta = np.zeros([n_samples, n_tasks])

    highest_d_obj = 0
    ws_size = min_ws_size

    tmpint = n_samples * n_tasks
    cdef double norm_Yfro = dnrm2(&tmpint, &Y[0, 0], &inc) ** 2
    cdef double[:] norms_X_col = np.empty(n_features)

    for j in range(n_features):
        norms_X_col[j] = dnrm2(&n_samples, &X[0, j], &inc)

    for t in range(max_iter):
        scal = mt_compute_alpha(Theta, Ksi, X)
        if scal < 0:
            raise ValueError("scal= %f < 0")

        if scal != 1.:
            for i in range(n_samples):
                for k in range(n_tasks):
                    Theta[i, k] = scal * Ksi[i, k] + (1 - scal) * Theta[i, k]
        else:
            # Theta = Ksi
            dcopy(&tmpint, &Ksi[0, 0], &inc, &Theta[0, 0], &inc)

        d_obj = mt_dual_value(alpha, n_samples, n_tasks, R,
                              Y, dual_scale, norm_Yfro)

        (Y, Theta, alpha, norm_Yfro)

        if d_obj > highest_d_obj:
            highest_d_obj = d_obj

        p_obj = mt_primal_value(alpha, n_samples, n_tasks, n_features,
                                R,  Beta)
        gap = p_obj - highest_d_obj
        gaps[t] = gap
        times[t] = time.time() - t0
        if verbose:
            print("----Iteration %d" % t)
            print("    Primal {:.10f}".format(p_obj))
            print("    Dual   {:.10f}".format(highest_d_obj))
            print("    Gap %.2e" % gap)

        if gap < tol:
            if gap < tol:
                print("Early exit, gap: %.2e < %.2e" % (gap, tol))
                break

        radius = sqrt(2. * gap) / alpha
        mt_set_feature_prios(n_samples, n_features, n_tasks,
                             Theta, X, &norms_X_col[0],
                             &prios[0], &disabled[0], radius)

        ws_size = 0
        for j in range(n_features):
            if screening:
                if disabled[j] == 0 and prios[j] > radius:
                    disabled[j] = 1
                    n_disabled += 1
                    if not is_zero(n_tasks, Beta[j]):
                        # TODO remove contribution of beta j
                        # fill beta j with 0
                        for k in range(n_tasks):
                            Beta[j, k] = 0
            if not is_zero(n_tasks, Beta[j]):
                prios[j] = -1.
                ws_size += 2

        if verbose and screening:
            print("%d disabled features" % n_disabled)
        if ws_size < min_ws_size:
            ws_size = min_ws_size
        if ws_size > n_features - n_disabled:
            ws_size = n_features - n_disabled

        C = np.argpartition(np.asarray(prios), ws_size)[:ws_size].astype(np.int32)
        C.sort()

        tol_inner = tol_ratio_inner * gap
        gram = compute_gram(n_samples, ws_size, C, X)

        if verbose:
            print("Solving subproblem with %d constraints" % len(C))

        sol_subpb = gram_mt_fast(X, Y, alpha, Beta, gram, XtY, C,
                                 norm_Yfro, tol_inner, max_updates, gap_spacing,
                                 strategy, batch_size)

        Beta_subpb, R, dual_scale, n_iter_ = sol_subpb[:4]
        Ksi = R / dual_scale  # = residuals /max(alpha, ||X^\top R||_\infty)



    else:
        print("!!!!!!!! Outer solver did not converge !!!!!!!!")

    return Beta, R,  np.asarray(gaps[:t + 1]), np.asarray(times[:t + 1])


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def gram_mt_fast(double[::1, :] X,
                 double[:, ::1] Y,
                 double alpha,
                 double[:, ::1] W_init,
                 double[::1, :] Q,
                 double[:, ::1] XtY,
                 double norm_Yfro,
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
    cdef int start
    cdef int stop

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
                                      dual_scale, norm_Yfro)
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
                    # compute block ST and put the result in w_ii_new
                    block_ST(n_tasks, &W[k, 0], &gradients[k, 0],
                             alpha, Q[k, k], &w_k_new[0])
                    norm_diff = norm_difference(n_tasks, &w_k_new[0], &W[k, 0])

                    if norm_diff >= best_norm_diff:
                        best_norm_diff = norm_diff
                        ii = k
                        # w_ii_new = w_k_new.copy():
                        dcopy(&n_tasks, &w_k_new[0], &inc, &w_ii_new[0], &inc)
            elif strategy == 2:  # cyclic
                ii = n_updates % n_features
                # compute block ST and put the result in w_ii_new
                block_ST(n_tasks, &W[ii, 0], &gradients[ii, 0],
                         alpha, Q[ii, ii], &w_ii_new[0])
            elif strategy == 3:
                # GS-rB
                start = (n_updates % nb_batch) * batch_size
                stop  = start + batch_size
                if stop > n_features:
                    stop = n_features
                ii = 0
                best_norm_diff = 0.
                for k in range(start, stop):
                    # compute block ST and put the result in w_k_new
                    block_ST(n_tasks, &W[k, 0], &gradients[k, 0],
                                      alpha, Q[k, k], &w_k_new[0])
                    norm_diff = norm_difference(n_tasks, &w_k_new[0], &W[k, 0])

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
