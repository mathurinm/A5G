from sklearn.datasets import load_svmlight_file
import time
from scipy import sparse
import numpy as np
from numpy.linalg import norm
import blitzl1
from a5g.lasso_fast import a5g_lasso_sparse
from a5g.utils import preprocess_data


if False:
    # download log1p.E2006.train.bz2 at
    # https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression.html
    with open("./data/log1p.E2006.train", 'rb') as f:
        X, y = load_svmlight_file(f, 4272227)
        X = sparse.csc_matrix(X)

        NNZ = np.diff(X.indptr)  # number of non zero elements per feature
        # keep only features with >=3 non zero values
        X_new = X[:, NNZ >= 3]

        sparse.save_npz("./data/finance_filtered", X_new)
        np.save("./data/finance_target", y)
else:
    X_new = sparse.load_npz("./data/finance_filtered.npz")
    y = np.load("./data/finance_target.npy")

preprocess = True
if preprocess:
    X_new, y = preprocess_data(X_new, y)


alpha_max = norm(X_new.T.dot(y), ord=np.inf)

n_features = X_new.shape[1]
alpha_div = 4
alpha = alpha_max / alpha_div

tol = 1e-8
max_iter = 40
max_updates = 50000
batch_size = 5
gap_spacing = 1000
tol_ratio_inner = 0.3
min_ws_size = 100

beta_init = np.zeros(n_features)
t0 = time.time()
a5g_res = a5g_lasso_sparse(X_new.data, X_new.indices, X_new.indptr, y, alpha,
                           beta_init, max_iter, gap_spacing, max_updates,
                           batch_size, tol_ratio_inner=tol_ratio_inner,
                           tol=tol, verbose=True, strategy=3,
                           min_ws_size=min_ws_size, screening=0)
dur_a5g = time.time() - t0
print("A5G time %.4f" % (dur_a5g))
beta = np.array(a5g_res[0])
gaps = a5g_res[2]
times = a5g_res[3]
print(beta[beta != 0])


from a5g.utils import primal, dual
R = y - X_new.dot(beta)

dual_scale = max(alpha, np.max(np.abs(X_new.T.dot(R))))
p_obj = primal(R, beta, alpha)
d_obj = dual(y, R / dual_scale, alpha, (y ** 2).sum())
print(p_obj - d_obj)
assert (p_obj - d_obj) < tol


t0 = time.time()
prob = blitzl1.LassoProblem(X_new, y)
blitzl1.set_use_intercept(True)
blitzl1.set_tolerance(tol)
sol = prob.solve(alpha)
print("Blitz time %.3f s" % (time.time() - t0))
beta_blitz = sol.x[sol.x != 0]
R = y - X_new.dot(sol.x)
p_obj_blitz = 0.5 * (R ** 2).sum() + alpha * norm(sol.x, ord=1)
