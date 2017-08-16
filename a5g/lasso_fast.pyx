import numpy as np
import time
cimport numpy as np
from scipy.linalg.cython_blas cimport ddot, dasum, daxpy, dnrm2, dcopy
from libc.math cimport fabs, sqrt, ceil
from libc.stdlib cimport rand, srand
cimport cython


cdef inline double fmax(double x, double y) nogil:
    return x if x > y else y


cdef inline double fmin(double x, double y) nogil:
    return y if x > y else x


cdef inline double fsign(double x) nogil :
    if x == 0.:
        return 0.
    elif x > 0.:
        return 1.
    else:
        return - 1.


cdef inline double ST(double u, double x) nogil:
    return fsign(x) * fmax(fabs(x) - u, 0.)


# for UT
def ST_c(double u, double x):
    return ST(u, x)


cdef double abs_max(int n, double * a) nogil:
    cdef int ii
    cdef double m = 0.
    cdef double d
    for ii in range(n):
        d = fabs(a[ii])
        if d > m:
            m = d
    return m


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef double primal_value(double alpha, int n_samples, double * R,
                         int n_features, double * w) nogil:
    cdef int inc = 1
    # regularization term: alpha ||w||_1
    cdef double p_obj = alpha * dasum(&n_features, w, &inc)
    p_obj += ddot(&n_samples, R, &inc, R, &inc) / 2.
    return p_obj


def primal_value_c(double alpha, int n_samples, double[:] R,
                    int n_features, double[:] w):
    return primal_value(alpha, n_samples, &R[0], n_features, &w[0])


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef double dual_value(double alpha, int n_samples, double * R, double *y,
                       double dual_scale, double norm_y2) nogil:
    cdef double d_obj = 0.
    cdef int i
    for i in range(n_samples):
        d_obj -= (y[i] / alpha - R[i] / dual_scale) ** 2
    d_obj *= 0.5 * alpha ** 2
    d_obj += 0.5 * norm_y2
    return d_obj


# used for UT
def dual_value_c(double alpha, int n_samples, double[:] R, double[:] y,
                     double dual_scale, double norm_y2):
    return dual_value(alpha, n_samples, &R[0], &y[0], dual_scale, norm_y2)


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
cdef double compute_scal(int n_samples, int n_features, double * theta,
                          double * ksi, double[::1, :] X,
                          int * disabled) nogil:
    cdef double Xj_ksi
    cdef double Xj_theta
    cdef double scal = 1.
    cdef int j
    cdef int inc = 1
    for j in range(n_features):
        if 1:
            Xj_ksi = fabs(ddot(&n_samples, &X[0, j], &inc, &ksi[0], &inc))
            if Xj_ksi > 1:
                Xj_theta = fabs(ddot(&n_samples, &X[0, j], &inc, &theta[0],
                                     &inc))
                if Xj_theta > 1.:
                    Xj_theta = 1
                scal = fmin(scal, (1. - Xj_theta) / (Xj_ksi - Xj_theta))
    return scal


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void set_feature_prios(int n_samples, int n_features, double * theta,
                            double[::1, :] X, double * norms_X_col,
                            double * prios, int * disabled,
                            double radius) nogil:
    """we set priorities of disabled features to radius + j
    This ensures that we don't include disabled features in the WS"""
    cdef int j
    cdef int inc = 1
    cdef double Xj_theta

    for j in range(n_features):
        if disabled[j] == 1:
            prios[j] = radius + j
        else:
            Xj_theta = ddot(&n_samples, &X[0, j], &inc, &theta[0], &inc)
            prios[j] = fabs(fabs(Xj_theta) - 1.) / norms_X_col[j]


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def a5g(double[::1, :] X,
        double[:] y,
        double alpha,
        double[:] beta_init,
        int max_iter,
        int gap_spacing,
        int max_updates,
        int batch_size,
        float tol_ratio_inner=0.3,
        float tol=1e-6,
        int min_ws_size=100,
        int screening=0,
        int strategy=3,
        int verbose=0,
        ):
    assert strategy in (1, 2, 3)
    cdef double t0 = time.time()

    cdef int n_samples = X.shape[0]
    cdef int n_features = X.shape[1]
    cdef double[:] beta = np.empty(n_features)
    cdef double[:] theta = np.zeros(n_samples)
    cdef double[:] ksi = np.zeros(n_samples)
    cdef int j  # features
    cdef int i  # samples
    cdef int t  # outer loop
    cdef int inc = 1
    cdef double tmp
    cdef int ws_size
    cdef double p_obj
    cdef double d_obj
    cdef double highest_d_obj
    cdef double gap
    cdef double radius  # for screening
    cdef double[:] prios = np.empty(n_features)
    cdef double[:] Xty = np.dot(X.T, y)

    cdef int[:] disabled = np.zeros(n_features, dtype=np.int32)
    cdef int n_disabled = 0

    for j in range(n_features):
        beta[j] = beta_init[j]

    for i in range(n_samples):
        ksi[i] = y[i] / alpha

    cdef double norm_y2 = dnrm2(&n_samples, &y[0], &inc) ** 2
    cdef double[:] norms_X_col = np.empty(n_features)

    for j in range(n_features):
        norms_X_col[j] = dnrm2(&n_samples, &X[0, j], &inc)

    cdef double[:] invnorm_Xcols_2 = np.empty(n_features)
    cdef double[:] alpha_invnorm_Xcols_2 = np.empty(n_features)

    for j in range(n_features):
        invnorm_Xcols_2[j] = 1. / norms_X_col[j] ** 2
        alpha_invnorm_Xcols_2[j] = alpha * invnorm_Xcols_2[j]


    cdef double[:] times = np.zeros(max_iter)
    cdef double[:] gaps = np.zeros(max_iter)
    cdef double[:] R = np.zeros(n_samples)

    cdef double[::1, :] gram

    for t in range(max_iter):
        scal = compute_scal(n_samples, n_features, &theta[0], &ksi[0],
                            X, &disabled[0])
        if scal == 1:  # ksi is feasible, set theta = ksi
            dcopy(&n_samples, &ksi[0], &inc, &theta[0], &inc)
        else:
            for i in range(n_samples):
                theta[i] = scal * ksi[i] + (1. - scal) * theta[i]

        dcopy(&n_samples, &y[0], &inc, &R[0], &inc)
        for j in range(n_features):
            if beta[j] == 0.:
                continue
            tmp = - beta[j]
            daxpy(&n_samples, &tmp, &X[0, j], &inc, &R[0], &inc)

        d_obj = 0.
        for i in range(n_samples):
            d_obj -= (y[i] / alpha - theta[i]) ** 2
        d_obj *= 0.5 * alpha ** 2
        d_obj += 0.5 * norm_y2
        if t == 0 or d_obj > highest_d_obj:
            highest_d_obj = d_obj

        p_obj = primal_value(alpha, n_samples, &R[0], n_features, &beta[0])
        gap = p_obj - highest_d_obj
        gaps[t] = gap
        if screening:
            radius = sqrt(2. * gap) / alpha
        times[t] = time.time() - t0

        if verbose:
            print("----Iteration %d" % t)
            print("    Primal {:.10f}".format(p_obj))
            print("    Dual   {:.10f}".format(highest_d_obj))
            print("    Gap %.2e" % gap)

        if gap < tol:
            print("Early exit, gap: %.2e < %.2e" % (gap, tol))
            break

        set_feature_prios(n_samples, n_features, &theta[0],
                          X, &norms_X_col[0], &prios[0], &disabled[0], radius)
        ws_size = 0
        for j in range(n_features):
            if screening:
                if disabled[j] == 0 and prios[j] > radius:
                    disabled[j] = 1
                    n_disabled += 1
                    if beta[j] != 0.:
                        daxpy(&n_samples, &beta[j], &X[0, j], &inc, &R[0], &inc)
                        beta[j] = 0.
            if beta[j] != 0.:
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
        # calling inner solver which will modify beta and R inplace
        dual_scale = gram_lasso_fast(n_samples, n_features, ws_size, X, y,
                                alpha, beta, R, C,
                                gram, Xty, invnorm_Xcols_2, alpha_invnorm_Xcols_2,
                                norm_y2, tol_inner, max_updates=max_updates,
                                gap_spacing=gap_spacing, strategy=strategy,
                                batch_size=batch_size, verbose=verbose)

        for i in range(n_samples):
            ksi[i] = R[i] / dual_scale

    return beta, R, np.asarray(gaps[:t + 1]), np.asarray(times[:t + 1])


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef double gram_lasso_fast(int n_samples, int n_features, int ws_size,
                   double[::1, :] X,
                   double[:] y,
                   double alpha,
                   double[:] beta,
                   double[:] R,
                   int[:] C,
                   double[::1, :] gram,
                   double[:] Xty,
                   double[:] invnorm_Xcols_2,
                   double[:] alpha_invnorm_Xcols_2,
                   double norm_y2,
                   double eps,
                   int max_updates,
                   int gap_spacing,
                   int strategy=3,
                   int batch_size=10,
                   int verbose=0,
                   int monitor_time=0,
                   int early_stop=1,
                   ):

    cdef int i # to iterate over samples.
    cdef int jj  # to iterate over features
    cdef int j
    cdef int k
    cdef int nb_batch = int(ceil(1.0 * ws_size / batch_size))
    cdef int start
    cdef int stop
    cdef int n_updates = 0
    cdef double beta_Cjj
    cdef double beta_Cjj_new
    cdef double beta_Cii_new
    cdef int inc = 1
    # gap related:
    cdef double gap
    cdef double[:] gaps = np.zeros(max_updates // gap_spacing, dtype=np.float64)
    cdef double dual_norm_XtR
    cdef double dual_scale
    cdef double d_obj
    cdef double highest_d_obj = 0. # d_obj is always >=0 so this gets replaced
    # at first d_obj computation. highest_d_obj corresponds to theta = 0.
    cdef double tmp

    cdef double update
    cdef double abs_update
    cdef double best_abs_update
    cdef double t0 = time.time()

    # initialize gradients: np.dot(gram, beta[C]) - Xty[C]
    cdef double[:] gradients = np.zeros(ws_size)
    for j in range(ws_size):
        if beta[C[j]] != 0.:
            daxpy(&ws_size, &beta[C[j]], &gram[0, j], &inc, &gradients[0], &inc)
        gradients[j] -= Xty[C[j]]

    for n_updates in range(max_updates):

        if (n_updates + 1) % gap_spacing == 0:
            # dual scale
            if monitor_time== 1:
                gaps[n_updates // gap_spacing] = time.time() - t0
            else:
                dual_norm_XtR = abs_max(ws_size, &gradients[0])
                dual_scale = fmax(alpha, dual_norm_XtR)

                for i in range(n_samples):
                    R[i] = y[i]
                for j in range(ws_size):
                    if beta[C[j]] == 0.:
                        continue
                    tmp = - beta[C[j]]
                    daxpy(&n_samples, &tmp, &X[0, C[j]], &inc, &R[0], &inc)


                d_obj = dual_value(alpha, n_samples, &R[0], &y[0], dual_scale,
                                   norm_y2)
                if n_updates == 1 or d_obj > highest_d_obj:
                    highest_d_obj = d_obj
                # we pass full beta and will ignore zero values
                gap = primal_value(alpha, n_samples, &R[0], n_features,
                                   &beta[0]) - highest_d_obj
                gaps[n_updates / gap_spacing] = gap
                if gap < eps and early_stop:
                    if verbose:
                        print("Inner: early exit at update %d, gap: %.2e < %.2e" % \
                            (n_updates, gap, eps))
                    return dual_scale


        # choose feature ii to update
        if strategy == 1:  # full greedy
            ii = 0
            best_abs_update = 0.
            for j in range(ws_size):
                k = C[j]
                update = - beta[k] + ST(alpha_invnorm_Xcols_2[k],
                                   beta[k] - gradients[j] * invnorm_Xcols_2[k])
                abs_update = fabs(update)
                if  j == 0 or abs_update > best_abs_update:
                    best_abs_update = abs_update
                    ii = j
                    beta_Cii_new = beta[C[ii]] + update
                    tmp = update

        elif strategy == 3:
            # Greedy Cyclic block (non random)
            start = (n_updates % nb_batch) * batch_size
            stop  = start + batch_size
            if stop > ws_size:
                stop = ws_size
            ii = start
            best_abs_update = 0.
            # j refers to the variable index in the WS. ii is relative too
            # k refers to the variable index in all features.
            for j in range(start, stop):
                k = C[j]
                update = - beta[k] + ST(alpha_invnorm_Xcols_2[k],
                                   beta[k] - gradients[j] * invnorm_Xcols_2[k])
                abs_update = fabs(update)
                if  j == start or abs_update > best_abs_update:
                    best_abs_update = abs_update
                    ii = j
                    beta_Cii_new = beta[C[ii]] + update
                    tmp = update

        elif strategy == 2:  # cyclic
            ii = n_updates % ws_size
            k = C[ii]
            update = - beta[C[ii]] + ST(alpha_invnorm_Xcols_2[k],
                               beta[k] - gradients[ii] * invnorm_Xcols_2[k])
            beta_Cii_new = beta[k] + update
            tmp = update

        # gradients += gram[:, ii] * (beta[C[ii]]_new - beta[C[ii]]_old)
        if tmp != 0.:
            daxpy(&ws_size, &tmp, &gram[0, ii], &inc, &gradients[0], &inc)
            beta[C[ii]] = beta_Cii_new

    # return correct residuals and dual_scale even if early exit wasnt triggered
    else:
        if verbose:
            print("!!!! inner solver did not converge !!!!")
        dual_norm_XtR = abs_max(ws_size, &gradients[0])
        dual_scale = fmax(alpha, dual_norm_XtR)

        for i in range(n_samples):
            R[i] = y[i]
        for j in range(ws_size):
            if beta[C[j]] == 0.:
                continue
            tmp = - beta[C[j]]
            daxpy(&n_samples, &tmp, &X[0, C[j]], &inc, &R[0], &inc)

    # if monitor_time == 1:
    #     return gaps  # which are times
    # if monitor_time == 2:
    #     return gaps  # which are gaps

    return dual_scale


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def compute_gram_sparse(int ws_size, int[:] C, double[:] X_data,
                        int[:] X_indices, int[:] X_indptr):
    cdef double[::1, :] gram = np.empty((ws_size, ws_size), order='F')
    cdef int j
    cdef int k
    cdef int j_ix
    cdef int k_ix

    cdef int startCj
    cdef int endCj
    cdef int startCk
    cdef int endCk
    cdef double tmp
    for j in range(ws_size):
        startCj = X_indptr[C[j]]
        endCj = X_indptr[C[j] + 1]

        for k in range(j + 1):
            startCk = X_indptr[C[k]]
            endCk = X_indptr[C[k] + 1]
            tmp = 0.
            j_ix = startCj
            k_ix = startCk
            while j_ix < endCj and k_ix < endCk:
                if X_indices[j_ix] ==  X_indices[k_ix]:
                    tmp += X_data[j_ix] * X_data[k_ix]
                    j_ix += 1
                    k_ix += 1
                elif X_indices[j_ix] < X_indices[k_ix]:
                    j_ix += 1
                else:  # X_indices[j_ix] > X_indices[k_ix]:
                    k_ix += 1

            gram[k, j] = tmp
            gram[j, k] = tmp
    return gram



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef double compute_scal_sparse(int n_features, double * theta,
                          double * ksi, double[:] X_data, int[:] X_indices,
                          int[:] X_indptr) nogil:
    cdef double Xj_ksi
    cdef double Xj_theta
    cdef double scal = 1.
    cdef int j
    cdef int i
    cdef int startptr
    cdef int endptr
    for j in range(n_features):
        Xj_ksi = 0.
        startptr = X_indptr[j]
        endptr = X_indptr[j + 1]
        for i in range(startptr, endptr):
            Xj_ksi += X_data[i] * ksi[X_indices[i]]
        Xj_ksi = fabs(Xj_ksi)

        if Xj_ksi > 1 + 1e-12: # avoid numerical errors
            Xj_theta = 0.
            for i in range(startptr, endptr):
                Xj_theta += X_data[i] * theta[X_indices[i]]
            Xj_theta = fabs(Xj_theta)
            if Xj_theta > 1.:
                Xj_theta = 1
            scal = fmin(scal, (1. - Xj_theta) / (Xj_ksi - Xj_theta))

    return scal


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void set_feature_prios_sparse(int n_features, double * theta,
                            double[:] X_data, int[:] X_indices,
                            int[:] X_indptr,
                            double * norms_X_col,
                            double * prios) nogil:
    cdef int j
    cdef int i
    cdef double Xj_theta
    cdef int startptr
    cdef int endptr

    for j in range(n_features):
        Xj_theta = 0
        startptr = X_indptr[j]
        endptr = X_indptr[j + 1]
        for i in range(startptr, endptr):
            Xj_theta += theta[X_indices[i]] * X_data[i]
        prios[j] = fabs(fabs(Xj_theta) - 1.) / norms_X_col[j]


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def a5g_sparse(double[:] X_data,
               int[:] X_indices,
               int[:] X_indptr,
               double[:] y,
               double alpha,
               double[:] beta_init,
               int max_iter,
               int gap_spacing,
               int max_updates,
               int batch_size,
               float tol_ratio_inner=0.3,
               float tol=1e-6,
               int min_ws_size=100,
               int screening=0,
               int strategy=3,
               int verbose=0
               ):
    assert strategy in (1, 2, 3)
    cdef double t0 = time.time()


    cdef int n_samples = y.shape[0]
    cdef int n_features = beta_init.shape[0]
    cdef double[:] beta = np.empty(n_features)
    cdef double[:] theta = np.zeros(n_samples)
    cdef double[:] ksi = np.zeros(n_samples)
    cdef int j  # features
    cdef int i  # samples
    cdef int ii
    cdef int t  # outer loop
    cdef int inc = 1
    cdef int startptr
    cdef int endptr
    cdef double tmp
    cdef double prov  # same as tmp (provisional)
    cdef int ws_size
    cdef double p_obj
    cdef double d_obj
    cdef double highest_d_obj
    cdef double gap
    cdef double[:] prios = np.empty(n_features)
    cdef double[:] Xty = np.empty(n_features)
    cdef double[:] norms_X_col = np.empty(n_features)

    # fill Xty and norms_X_col
    for j in range(n_features):
        startptr = X_indptr[j]
        endptr = X_indptr[j + 1]
        tmp = 0.
        prov = 0.
        for i in range(startptr, endptr):
            prov += X_data[i] ** 2
            tmp += y[X_indices[i]] * X_data[i]

        Xty[j] = tmp
        norms_X_col[j] = sqrt(prov)

    for j in range(n_features):
        beta[j] = beta_init[j]

    for i in range(n_samples):
        ksi[i] = y[i] / alpha

    cdef double norm_y2 = dnrm2(&n_samples, &y[0], &inc) ** 2
    cdef double[:] invnorm_Xcols_2 = np.empty(n_features)
    cdef double[:] alpha_invnorm_Xcols_2 = np.empty(n_features)

    for j in range(n_features):
        invnorm_Xcols_2[j] = 1. / norms_X_col[j] ** 2
        alpha_invnorm_Xcols_2[j] = alpha * invnorm_Xcols_2[j]


    cdef double[:] times = np.zeros(max_iter)
    cdef double[:] gaps = np.zeros(max_iter)
    cdef double[:] R = np.zeros(n_samples)

    cdef double[::1, :] gram

    for t in range(max_iter):
        scal = compute_scal_sparse(n_features, &theta[0], &ksi[0],
                            X_data, X_indices, X_indptr)
        if scal == 1:  # ksi is feasible, set theta = ksi
            print("Feasible ksi, complicated scaling might not be needed")
            dcopy(&n_samples, &ksi[0], &inc, &theta[0], &inc)
        elif scal == 0.:
            print("WARNING scal=0, d_obj will stay the same, \
                  this should not happen ")

        else:
            for i in range(n_samples):
                theta[i] = scal * ksi[i] + (1. - scal) * theta[i]

        # compute residuals :
        dcopy(&n_samples, &y[0], &inc, &R[0], &inc)
        for j in range(n_features):
            if beta[j] == 0.:
                continue
            else:
                startptr = X_indptr[j]
                endptr = X_indptr[j + 1]
                for ii in range(startptr, endptr):
                    R[X_indices[ii]] -= beta[j] * X_data[ii]

        d_obj = 0.
        for i in range(n_samples):
            d_obj -= (y[i] / alpha - theta[i]) ** 2
        d_obj *= 0.5 * alpha ** 2
        d_obj += 0.5 * norm_y2
        if t == 0 or d_obj > highest_d_obj:
            highest_d_obj = d_obj

        p_obj = primal_value(alpha, n_samples, &R[0], n_features, &beta[0])
        gap = p_obj - highest_d_obj
        gaps[t] = gap
        times[t] = time.time() - t0

        if verbose:
            print("----Iteration %d" % t)
            print("    Primal {:.10f}".format(p_obj))
            print("    Dual   {:.10f}".format(highest_d_obj))
            print("    Gap %.2e" % gap)

        if gap < tol:
            print("Early exit, gap: %.2e < %.2e" % (gap, tol))
            break

        set_feature_prios_sparse(n_features, &theta[0],
                          X_data, X_indices, X_indptr,
                          &norms_X_col[0], &prios[0])
        ws_size = 0
        for j in range(n_features):
            if beta[j] != 0:
                prios[j] = -1.
                ws_size += 2

        if ws_size < min_ws_size:
            ws_size = min_ws_size
        if ws_size > n_features:
            ws_size = n_features

        C = np.argpartition(np.asarray(prios), ws_size)[:ws_size].astype(np.int32)
        C.sort()
        # print(C)
        tol_inner = tol_ratio_inner * gap
        gram = compute_gram_sparse(ws_size, C, X_data, X_indices, X_indptr)

        if verbose:
            print("Solving subproblem with %d constraints" % len(C))
        # calling inner solver which will modify beta and R inplace
        dual_scale = gram_lasso_fast_sparse(n_samples, n_features, ws_size,
                                X_data, X_indices, X_indptr,
                                y,
                                alpha, beta, R, C,
                                gram, Xty, invnorm_Xcols_2, alpha_invnorm_Xcols_2,
                                norm_y2, tol_inner, max_updates=max_updates,
                                gap_spacing=gap_spacing, strategy=strategy,
                                batch_size=batch_size, verbose=verbose)

        for i in range(n_samples):
            ksi[i] = R[i] / dual_scale

    return beta, R, np.asarray(gaps[:t + 1]), np.asarray(times[:t + 1])


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef double gram_lasso_fast_sparse(int n_samples, int n_features, int ws_size,
                   double[:] X_data,
                   int[:] X_indices,
                   int[:] X_indptr,
                   double[:] y,
                   double alpha,
                   double[:] beta,
                   double[:] R,
                   int[:] C,
                   double[::1, :] gram,
                   double[:] Xty,
                   double[:] invnorm_Xcols_2,
                   double[:] alpha_invnorm_Xcols_2,
                   double norm_y2,
                   double eps,
                   int max_updates,
                   int gap_spacing,
                   int strategy=3,
                   int batch_size=10,
                   int verbose=0
                   ):

    cdef int startptr
    cdef int endptr

    cdef int i # to iterate over samples.
    cdef int ii
    cdef int jj  # to iterate over features
    cdef int j
    cdef int k
    cdef int nb_batch = int(ceil(1.0 * ws_size / batch_size))
    cdef int start
    cdef int stop
    cdef int n_updates = 0
    cdef double beta_Cj
    cdef double beta_Cjj
    cdef double beta_Cjj_new
    cdef double beta_Cii_new
    cdef int inc = 1
    # gap related:
    cdef double gap
    cdef double[:] gaps = np.zeros(max_updates // gap_spacing, dtype=np.float64)
    cdef double dual_norm_XtR
    cdef double dual_scale
    cdef double d_obj
    cdef double highest_d_obj = 0. # d_obj is always >=0 so this gets replaced
    # at first d_obj computation. highest_d_obj corresponds to theta = 0.
    cdef double tmp

    cdef double update
    cdef double abs_update
    cdef double best_abs_update

    # initialize gradients: np.dot(gram, beta[C]) - Xty[C]
    cdef double[:] gradients = np.zeros(ws_size)
    for j in range(ws_size):
        if beta[C[j]] != 0.:
            daxpy(&ws_size, &beta[C[j]], &gram[0, j], &inc, &gradients[0], &inc)
        gradients[j] -= Xty[C[j]]

    for n_updates in range(max_updates):

        if n_updates % gap_spacing == 1:
            # dual scale
            dual_norm_XtR = abs_max(ws_size, &gradients[0])
            dual_scale = fmax(alpha, dual_norm_XtR)
            for i in range(n_samples):
                R[i] = y[i]
            for j in range(ws_size):
                beta_Cj = beta[C[j]]
                if beta_Cj == 0.:
                    continue
                else:
                    startptr = X_indptr[C[j]]
                    endptr = X_indptr[C[j] + 1]
                    for ii in range(startptr, endptr):
                        R[X_indices[ii]] -= beta_Cj * X_data[ii]


            d_obj = dual_value(alpha, n_samples, &R[0], &y[0], dual_scale,
                               norm_y2)
            if n_updates == 1 or d_obj > highest_d_obj:
                highest_d_obj = d_obj
            # we pass full beta and will ignore zero values
            gap = primal_value(alpha, n_samples, &R[0], n_features,
                               &beta[0]) - highest_d_obj
            gaps[n_updates / gap_spacing] = gap
            if gap < eps:
                if verbose:
                    print("Inner: early exit at update %d, gap: %.2e < %.2e" % \
                        (n_updates, gap, eps))
                return dual_scale


        # choose feature ii to update
        if strategy == 1:
            # full greedy
            for j in range(ws_size):
                k = C[j]
                update = - beta[k] + ST(alpha_invnorm_Xcols_2[k],
                                   beta[k] - gradients[j] * invnorm_Xcols_2[k])
                abs_update = fabs(update)
                if  j == 0 or abs_update > best_abs_update:
                    best_abs_update = abs_update
                    ii = j
                    beta_Cii_new = beta[C[ii]] + update
                    tmp = update

        elif strategy == 3:
            # Greedy Cyclic block (non random)
            start = (n_updates % nb_batch) * batch_size
            stop  = start + batch_size
            if stop > ws_size:
                stop = ws_size
            ii = start
            best_abs_update = 0.
            # j refers to the variable index in the WS. ii is relative too
            # k refers to the variable index in all features.
            for j in range(start, stop):
                k = C[j]
                update = - beta[k] + ST(alpha_invnorm_Xcols_2[k],
                                   beta[k] - gradients[j] * invnorm_Xcols_2[k])
                abs_update = fabs(update)
                if  j == start or abs_update > best_abs_update:
                    best_abs_update = abs_update
                    ii = j
                    beta_Cii_new = beta[C[ii]] + update
                    tmp = update

        elif strategy == 2:  # cyclic
            ii = n_updates % ws_size
            k = C[ii]
            update = - beta[k] + ST(alpha_invnorm_Xcols_2[k],
                               beta[k] - gradients[ii] * invnorm_Xcols_2[k])
            beta_Cii_new = beta[k] + update
            tmp = update


        # gradients += gram[:, ii] * (beta[C[ii]]_new - beta[C[ii]]_old)
        if tmp != 0.:
            daxpy(&ws_size, &tmp, &gram[0, ii], &inc, &gradients[0], &inc)
            beta[C[ii]] = beta_Cii_new

    # return correct residuals and dual_scale even if early exit wasnt triggered
    else:
        if verbose:
            print("!!!! inner solver did not converge !!!!")
            print("Last computed gap: %.2e" % gap)
        dual_norm_XtR = abs_max(ws_size, &gradients[0])
        dual_scale = fmax(alpha, dual_norm_XtR)

        for i in range(n_samples):
            R[i] = y[i]
        for j in range(ws_size):
            beta_Cj = beta[C[j]]
            if beta_Cj == 0.:
                continue
            else:
                startptr = X_indptr[C[j]]
                endptr = X_indptr[C[j] + 1]
                for ii in range(startptr, endptr):
                    R[X_indices[ii]] -= beta_Cj * X_data[ii]

    return dual_scale
