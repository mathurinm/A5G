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


cdef inline double fmin(double x, double y) nogil:
    return x if x < y else y


@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline int is_zero(int length, double * vector) nogil:
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
        # Beta is C ordered
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
    # w_new = w - grad_k / norm_Xk2
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
cdef double norm_difference(int length, double * u, double * v):
    """Compute ||u - v||_2"""
    cdef double nrm = 0.
    cdef int i
    for i in range(length):
        nrm += (v[i] - u[i]) ** 2
    return sqrt(nrm)


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
                               double * prios, int * disabled, double radius,
                               double * Xj_Theta):
    cdef int j  # features
    cdef int t  # tasks
    cdef int inc = 1
    cdef double tmp
    for j in range(n_features):
        if disabled[j] == 1:
            prios[j] = radius + j
        else:
            for t in range(n_tasks):
                # Theta is not fortran
                Xj_Theta[t] = ddot(&n_samples, &X[0, j], &inc, &Theta[0, t], &n_tasks)
            tmp = dnrm2(&n_tasks, &Xj_Theta[0], &inc)
        prios[j] = (1 - tmp) / norms_X_col[j]


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef double mt_compute_alpha(int n_samples, int n_features, int n_tasks,
                             double[:, ::1] Theta, double[:, ::1] Ksi,
                             double[::1, :] X, int * disabled,
                             double * Xj_Ksi, double * Xj_Theta):
    cdef int j  # features
    cdef int t  # tasks
    cdef double scal = 1.
    cdef int inc = 1
    cdef double tmp

    for j in range(n_features):
        if disabled[j]:
            continue
        for t in range(n_tasks):
            # Ksi is not Fortran
            Xj_Ksi[t] = ddot(&n_samples, &X[0, j], &inc,
                            &Ksi[0, t], &n_tasks)
        if dnrm2(&n_tasks, &Xj_Ksi[0], &inc) > 1. + 1e-12:
            Xj_Theta[t] = ddot(&n_samples, &X[0, j], &inc,
                              &Theta[0, t], &n_tasks)
            tmp = dnrm2(&n_tasks, &Xj_Theta[0], &inc)
            # Theta should be feasible:
            if tmp > 1.:
                tmp = 1
            scal = fmin(scal, (1. - tmp) / norm_difference(n_tasks, &Xj_Theta[0], &Xj_Ksi[0]))
    return scal


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def a5g_mt(double[::1, :] X,
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
    cdef int t  # tasks
    cdef int it  # outer loop iterations
    cdef int ws_size
    cdef double p_obj
    cdef double d_obj
    cdef double dual_scale
    cdef double gap
    cdef double radius
    cdef double[:] prios = np.empty(n_features)
    cdef double[:, ::1] XtY = np.dot(X.T, Y)
    cdef double scal
    cdef double tmp
    cdef int tmpint

    cdef double[:] gaps = np.zeros(max_iter)
    cdef double[:] times = np.zeros(max_iter)
    cdef int[:] disabled = np.zeros(n_features, dtype=np.int32)
    cdef int n_disabled = 0

    cdef double[:, ::1] Beta = np.empty([n_features, n_tasks])
    tmpint = n_features * n_tasks
    dcopy(&tmpint, &Beta_init[0, 0], &inc, &Beta[0, 0], &inc)

    cdef double[:, ::1] R = Y - np.dot(X, Beta)
    cdef double[:, ::1] Ksi = np.zeros([n_samples, n_tasks])
    tmp = 1. / alpha
    tmpint = n_samples * n_tasks
    daxpy(&tmpint, &tmp, &R[0, 0], &inc, &Ksi[0, 0], &inc)
    cdef double[:, ::1] Theta = np.zeros([n_samples, n_tasks])

    # preallocating vectors used in mt_set_feature_prios and mt_compute_alpha
    cdef double[:] Xj_Ksi = np.empty(n_tasks)
    cdef double[:] Xj_Theta = np.empty(n_tasks)

    highest_d_obj = 0
    ws_size = min_ws_size

    tmpint = n_samples * n_tasks
    cdef double norm_Yfro = dnrm2(&tmpint, &Y[0, 0], &inc) ** 2
    cdef double[:] norms_X_col = np.empty(n_features)
    cdef double[:] invnorm_Xcols_2 = np.empty(n_features)
    cdef double[:] alpha_invnorm_Xcols_2 = np.empty(n_features)

    for j in range(n_features):
        norms_X_col[j] = dnrm2(&n_samples, &X[0, j], &inc)
        invnorm_Xcols_2[j] = 1. / norms_X_col[j] ** 2
        alpha_invnorm_Xcols_2[j] = alpha / norms_X_col[j] ** 2

    for it in range(max_iter):
        scal = mt_compute_alpha(n_samples, n_features, n_tasks,
                                Theta, Ksi, X, &disabled[0], &Xj_Ksi[0],
                                &Xj_Theta[0])
        if scal < 0:
            raise ValueError("scal= %f < 0")

        if scal == 1:
            if verbose:
                print("Feasible Ksi, complicated scaling might not be needed")
            dcopy(&tmpint, &Ksi[0, 0], &inc, &Theta[0, 0], &inc)
        elif scal <= 0:
            print("WARNING scal=0, d_obj will stay the same, \
                  this should not happen ")
        else:
            tmpint = n_samples * n_tasks
            # Theta = scal * Ksi + (1 - scal) * Theta
            tmp = 1. - scal
            dscal(&tmpint, &tmp, &Theta[0, 0], &inc)
            daxpy(&tmpint, &scal, &Ksi[0, 0], &inc, &Theta[0, 0], &inc)


        # Compute dual rescaling to make R feasible
        # use preallocated vector Xj_Theta for this (in fact it contains Xj_R)
        dual_norm_XtR = 0.
        for j in range(n_features):
            for t in range(n_tasks):
                Xj_Theta[t] = ddot(&n_samples, &X[0, j], &inc, &R[0, t], &n_tasks)
            tmp = dnrm2(&n_tasks, &Xj_Theta[0], &inc)
            if tmp > dual_norm_XtR:
                dual_norm_XtR = tmp

        dual_scale = max(alpha, dual_norm_XtR)

        d_obj = mt_dual_value(alpha, n_samples, n_tasks, R,
                              Y, dual_scale, norm_Yfro)


        if d_obj > highest_d_obj:
            highest_d_obj = d_obj

        p_obj = mt_primal_value(alpha, n_samples, n_tasks, n_features,
                                R,  Beta)
        gap = p_obj - highest_d_obj
        gaps[it] = gap
        times[it] = time.time() - t0
        if verbose:
            print("----Iteration %d" % it)
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
                             &prios[0], &disabled[0], radius, &Xj_Theta[0])

        ws_size = 0
        for j in range(n_features):
            if screening:
                if disabled[j] == 0 and prios[j] > radius:
                    disabled[j] = 1
                    n_disabled += 1
                    if not is_zero(n_tasks, &Beta[j, 0]):
                        # TODO remove contribution of beta j
                        # fill beta j with 0
                        for k in range(n_tasks):
                            Beta[j, k] = 0
            if not is_zero(n_tasks, &Beta[j, 0]):
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

        # Beta and R are modified in place by gram_mt_fast()
        dual_scale = gram_mt_fast(n_samples, n_features, n_tasks, ws_size,
                                 X, Y, alpha, Beta, R, C, gram, XtY, invnorm_Xcols_2,
                                 alpha_invnorm_Xcols_2,
                                 norm_Yfro, tol_inner, max_updates, gap_spacing,
                                 strategy, batch_size, verbose=verbose)

        # Ksi = R / dual_scale
        tmpsum = 1. / dual_scale
        tmpint = n_samples * n_tasks
        dcopy(&tmpint, &R[0, 0], &inc, &Ksi[0, 0], &inc)
        dscal(&tmpint, &tmpsum, &Ksi[0, 0], &inc)

    else:
        print("!!!!!!!! Outer solver did not converge !!!!!!!!")

    return np.asarray(Beta), np.asarray(R), np.asarray(gaps[:it + 1]), np.asarray(times[:it + 1])


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef double gram_mt_fast(int n_samples,
                  int n_features,
                  int n_tasks,
                  int ws_size,
                  double[::1, :] X,
                  double[:, ::1] Y,
                  double alpha,
                  double[:, ::1] Beta,
                  double[:, ::1] R,
                  int[:] C,
                  double[::1, :] gram,
                  double[:, ::1] XtY,
                  double[:] invnorm_Xcols_2,
                  double[:] alpha_invnorm_Xcols_2,
                  double norm_Yfro,
                  double eps,
                  int max_updates,
                  int gap_spacing,
                  int strategy=3,
                  int batch_size=10,
                  int verbose=0,
                  seed=0):

    cdef int i
    cdef int ii
    cdef int j
    cdef int k
    cdef int nb_batch = int(ceil(1.0 * ws_size / batch_size))
    cdef int start
    cdef int stop
    cdef int n_updates
    cdef double[:] Beta_Cii_new = np.zeros(n_tasks)
    cdef double[:] Beta_k_new = np.zeros(n_tasks)
    cdef int inc = 1
    cdef double minus_one = - 1.  # for cblas...
    cdef double one = 1  # ditto
    cdef double tmpsum

    # gap related
    cdef double gap
    cdef double XtR_axis1norm  # same as in sklearn: norm(XtR, axis=1)[j]
    cdef double dual_norm_XtR
    cdef double dual_scale
    cdef double d_obj
    cdef double highest_d_obj = 0.

    cdef double[:] tmp = np.zeros(n_tasks, dtype=np.float64)
    cdef double[:] update = np.zeros(n_tasks, dtype=np.float64)
    cdef double norm_diff = 0.
    cdef double best_norm_diff = 0.

    # initialize gradients: np.dot(gram, beta[C]) - Xty[C]
    cdef double[:, ::1] gradients = np.empty([ws_size, n_tasks])
    # gram is fortran and symmetric so we use columns instead of row
    for j in range(ws_size):
        for t in range(n_tasks):
            tmpsum = - XtY[C[j], t]
            for k in range(ws_size):
                tmpsum += gram[j, k] * Beta[C[k], t]
            gradients[j, t] = tmpsum

    assert batch_size <= ws_size
    assert strategy in (1, 2, 3)
    srand(seed)

    for n_updates in range(max_updates):
        if (n_updates + 1) % gap_spacing == 0:
            # dual scale
            dual_norm_XtR = 0.
            for j in range(ws_size):
                XtR_axis1norm = dnrm2(&n_tasks, &gradients[j, 0], &inc)
                if XtR_axis1norm > dual_norm_XtR:
                    dual_norm_XtR = XtR_axis1norm

            dual_scale = fmax(alpha, dual_norm_XtR)
            # R = Y - np.dot(X, Beta)
            for i in range(n_samples):
                for t in range(n_tasks):
                    R[i, t] = Y[i, t] - ddot(&n_features,
                                             &X[i, 0], &n_samples,
                                             &Beta[0, t], &n_tasks)

            d_obj = mt_dual_value(alpha, n_samples, n_tasks, R, Y,
                                  dual_scale, norm_Yfro)
            if n_updates == (gap_spacing - 1) or d_obj > highest_d_obj:
                highest_d_obj = d_obj

            gap = mt_primal_value(alpha, n_samples, n_tasks, n_features, R,
                                  Beta) - highest_d_obj
            if gap < eps:
                if verbose:
                    print("Inner: early exit at update %d, gap: %.2e < %.2e" % \
                        (n_updates, gap, eps))
                return dual_scale


        # choose feature ii to update
        if strategy == 1: # full greedy
            ii = 0
            best_norm_diff = 0.
            for j in range(ws_size):
                k = C[j]
                # compute block ST and put the result in w_k_new
                block_ST(n_tasks, &Beta[k, 0], &gradients[j, 0],
                         alpha, gram[j, j], &Beta_k_new[0])
                norm_diff = norm_difference(n_tasks, &Beta_k_new[0],
                                            &Beta[k, 0])

                if j == 0 or norm_diff > best_norm_diff:
                    best_norm_diff = norm_diff
                    ii = j
                    # Beta_Cii_new = Beta_k_new.copy():
                    dcopy(&n_tasks, &Beta_k_new[0], &inc, &Beta_Cii_new[0], &inc)

        elif strategy == 2:  # cyclic
            ii = n_updates % ws_size
            k = C[ii]
            # compute block ST and put the result in Beta_Cii_new
            block_ST(n_tasks, &Beta[k, 0], &gradients[ii, 0],
                     alpha, gram[ii, ii], &Beta_Cii_new[0])

        elif strategy == 3:
            # GS-rB
            start = (n_updates % nb_batch) * batch_size
            stop  = start + batch_size
            if stop > ws_size:
                stop = ws_size
            ii = start
            best_norm_diff = 0.
            for j in range(start, stop):
                k = C[j]
                # compute block ST and put the result in w_k_new
                block_ST(n_tasks, &Beta[k, 0], &gradients[j, 0],
                                  alpha, gram[j, j], &Beta_k_new[0])
                norm_diff = norm_difference(n_tasks, &Beta_k_new[0],
                                            &Beta[k, 0])

                if j == start or norm_diff > best_norm_diff:
                    best_norm_diff = norm_diff
                    ii = j
                    # Beta_Cii_new = Beta_k_new.copy():
                    dcopy(&n_tasks, &Beta_k_new[0], &inc,
                          &Beta_Cii_new[0], &inc)

        # tmp = Beta_Cii_new - Beta[C[ii], :] in two steps
        # tmp = Beta_Cii_new
        dcopy(&n_tasks, &Beta_Cii_new[0], &inc, &tmp[0], &inc)
        # tmp -= Beta[C[ii], :]
        daxpy(&n_tasks, &minus_one, &Beta[C[ii], 0], &inc, &tmp[0], &inc)

        if not is_zero(n_tasks, &tmp[0]): # if tmp is not 0
            # update Beta[C[ii], :]
            dcopy(&n_tasks, &Beta_Cii_new[0], &inc, &Beta[C[ii], 0], &inc)

            # update all gradients
            for k in range(ws_size):
                for t in range(n_tasks):
                    gradients[k, t] += gram[k, ii] * tmp[t]


    else:
        # return correct residuals and dual_scale
        # even if Solver did not converge
        # TODO
        print('!!!! Inner solver did not converge !!!!')

    return dual_scale
