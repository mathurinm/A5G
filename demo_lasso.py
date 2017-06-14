import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets.mldata import fetch_mldata
from a5g.lasso_fast import a5g


###############################################################################
# Leukemia data
###############################################################################
dataset_id = 'leukemia'
leuk = fetch_mldata('leukemia')
X = np.asfortranarray(leuk.data)
y = leuk.target.astype(float)
y /= np.linalg.norm(y)
n_samples, n_features = X.shape
alpha_max = np.max(np.abs(np.dot(X.T, y)))
alpha_div = 50
alpha = alpha_max / alpha_div

max_updates = 10 ** 6  # maximum number of updates in the inner loop
gap_spacing = 10 ** 3  # number of updates between gap computation in inner solver

tol = 1e-6  # duality gap criterion for outer problem
batch_size = 10  # number of features per batch for GS-rB strategy
p0 = 100  # size of first working set
tol_ratio_inner = 0.3  # \underbar{\epsilon} in the paper
max_iter = 30  # max number of outer iterations
strategy = 3  # we use GS-rB: greedy on cyclic batches
verbose = True


# average repetition to account for timing variability
repeats = 5
all_times = [[] for _ in range(repeats)]

for i in range(repeats):
    print("Repetition %d" % i)
    t0 = time.time()
    beta_init = np.zeros(n_features)
    output = a5g(X, y, alpha, beta_init,
                 max_iter, gap_spacing, max_updates, batch_size,
                 tol_ratio_inner=tol_ratio_inner, tol=tol, verbose=verbose,
                 strategy=strategy, screening=1)
    dur_a5g = time.time() - t0
    print("A5G time %.5f" % dur_a5g)

    beta = np.asarray(output[0])
    gaps = np.asarray(output[2])
    all_times[i] = np.asarray(output[3])

times = np.array(all_times)
times = np.mean(times, axis=0)

# check that we have indeed converged
from a5g.utils import primal, dual
R = y - np.dot(X, beta)
dual_scale = max(alpha, np.max(np.abs(np.dot(X.T, R))))
p_obj = primal(R, beta, alpha)
d_obj = dual(y, R / dual_scale, alpha, np.dot(y, y))
assert 0 <= (p_obj - d_obj) < 1e-6


plt.semilogy(times, gaps)
plt.xlabel("Time (s)")
plt.ylabel("Duality gap")
plt.show()
