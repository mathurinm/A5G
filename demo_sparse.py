import numpy as np
import time
from a5g.lasso_fast import a5g_sparse
from scipy import sparse

n_samples = 1608
n_features = 10000

X = np.random.randn(n_samples, n_features)
X[np.random.uniform(size=(n_samples, n_features)) < 0.9] = 0
X = sparse.csc_matrix(X)

y = np.random.randn(n_samples)
alpha_max = np.max(np.abs(X.T.dot(y)))

beta_init = np.zeros(n_features)
alpha_div = 10
alpha = alpha_max / alpha_div

tol = 1e-6

max_iter = 20
max_updates = 50000
batch_size = 10
gap_spacing = 1000
tol_ratio_inner = 0.2


t0 = time.time()
test = a5g_sparse(X.data, X.indices, X.indptr, y, alpha, beta_init,
                  max_iter, gap_spacing, max_updates, batch_size,
                  tol_ratio_inner=tol_ratio_inner, tol=tol, verbose=True,
                  strategy=3)
dur_a5g = time.time() - t0
print("A5G time %.4f" % (dur_a5g))


beta = np.array(test[0])

from a5g.utils import primal, dual
R = y - X.dot(beta)

dual_scale = max(alpha, np.max(np.abs(X.T.dot(R))))
p_obj = primal(R, beta, alpha)
d_obj = dual(y, R / dual_scale, alpha, (y ** 2).sum())
assert 0 < (p_obj - d_obj) < tol
