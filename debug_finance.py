from sklearn.datasets import load_svmlight_file
import time
from scipy import sparse
import numpy as np
from a5g.lasso_fast import a5g_sparse
from numpy.linalg import norm
from sklearn.preprocessing import normalize


if False:
    with open("./data/log1p.E2006.train", 'rb') as f:
        X, y = load_svmlight_file(f, 4272227)
        X = sparse.csc_matrix(X)

        NNZ = np.diff(X.indptr)
        X_new = X[:, NNZ >= 3]

        sparse.save_npz("finance_filtered", X_new)
        np.save("finance_target", y)
else:
    X_new = sparse.load_npz("finance_filtered.npz")
    y = np.load("finance_target.npy")


alpha_max = norm(X_new.T.dot(y), ord=np.inf)

n_features = X_new.shape[1]
alpha_div = 2
alpha = alpha_max / alpha_div

tol = 1e-6

max_iter = 5
max_updates = 50000
batch_size = 10
gap_spacing = 1000
tol_ratio_inner = 0.3

beta_init = np.zeros(n_features)
t0 = time.time()
test = a5g_sparse(X_new.data, X_new.indices, X_new.indptr, y, alpha, beta_init,
                  max_iter, gap_spacing, max_updates, batch_size,
                  tol_ratio_inner=tol_ratio_inner, tol=tol, verbose=True,
                  strategy=2)
dur_a5g = time.time() - t0
print("A5G time %.4f" % (dur_a5g))
beta = np.array(test[0])
print(beta[beta != 0])


from a5g.utils import primal, dual
R = y - X_new.dot(beta)

dual_scale = max(alpha, np.max(np.abs(X_new.T.dot(R))))
p_obj = primal(R, beta, alpha)
d_obj = dual(y, R / dual_scale, alpha, (y ** 2).sum())
print(p_obj - d_obj)
assert (p_obj - d_obj) < tol


from sklearn.linear_model import Lasso
clf = Lasso(alpha=alpha / len(y))
clf.fit(X_new, y)



from a5g.lasso_fast import compute_gram_sparse
C = np.random.choice(X_new.shape[1], 20, replace=False).astype(np.int32)
gram_test = compute_gram_sparse(len(C), C, X_new.data, X_new.indices,
                                X_new.indptr)
gram_test = np.array(gram_test)
a = gram_test
b = (X_new[:, C].T.dot(X_new[:, C])).toarray()
np.testing.assert_allclose(a, b)


c = np.random.uniform(-1, 1, size=(10, 10))
c[np.random.uniform(size=c.shape) >0.4] = 0
c = sparse.csc_matrix(c)
C = np.array([0, 1, 2]).astype(np.int32)
np.array(compute_gram_sparse(len(C), C, c.data, c.indices,
                                c.indptr))
c[:, C].T.dot(c[:, C]).toarray()
