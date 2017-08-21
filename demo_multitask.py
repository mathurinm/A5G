import numpy as np
from numpy.linalg import norm
from a5g import a5g_mt
from a5g.utils import mt_primal, mt_dual, norml2inf

X = np.load("./data/G.npy")
Y = np.ascontiguousarray(np.load("./data/M.npy"))
X = np.asfortranarray(X[:, 2::3])  # use fixed orientation
n_samples, n_features = X.shape
n_tasks = Y.shape[1]

assert np.isfortran(X)
Y = Y / norm(Y, ord='fro')
alpha_max = np.max(norm(np.dot(X.T, Y), axis=1, ord=2))
alpha_div = 10
alpha = alpha_max / alpha_div


Beta_init = np.zeros([n_features, n_tasks])
res = a5g_mt(X, Y, alpha, Beta_init, max_iter=30, strategy=3, min_ws_size=20,
             max_updates=50000, screening=1)
Beta = res[0]
R = res[1]

support = np.where(Beta.any(axis=1))[0]

p_obj = mt_primal(R, Beta, alpha)
print("p_obj: %.10f" % p_obj)

Theta = R.copy()
dual_scale = max(alpha, norml2inf(np.dot(X.T, R)))
Theta /= dual_scale

d_obj = mt_dual(Y, Theta, alpha, (Y ** 2).sum())
print("d_obj: %.10f" % d_obj)
