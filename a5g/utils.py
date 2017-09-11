import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
from numpy.linalg import norm
from sklearn import preprocessing


def primal(R, beta, alpha):
    return 0.5 * norm(R, ord=2) ** 2 +\
        alpha * norm(beta, ord=1)


def dual(y, theta, alpha, norm_y2):
    return 0.5 * norm_y2 - \
        0.5 * alpha ** 2 * norm(y / alpha - theta, ord=2) ** 2


def ST(u, x):
    return np.sign(x) * np.maximum(np.abs(x) - u, 0.)


def compute_alpha(theta, ksi, X):
    X_ksi = np.abs(np.dot(X.T, ksi))

    # Constraint violated by ksi
    unmet = np.where(X_ksi > 1.)[0]
    if len(unmet) == 0:
        return 1.

    X_theta = np.abs(np.dot(X.T[unmet], theta))
    # [MM] I am trying this because theta is feasible so it
    # should not change anything, nevertheless I got a value of
    #  1.0000000000004 once which gave  a < 0 alpha:
    X_theta = np.minimum(1., X_theta)
    # compute alpha_j only for constraints violated by ksi:
    alphas = (1. - X_theta) / (X_ksi[unmet] - X_theta)

    return np.min(alphas)


def feature_prios(theta, X, norms_X_col):
    """computes distances of dual point theta to each constraint
    defined by a row of X"""
    XTtheta = np.dot(X.T, theta)

    return np.abs(np.abs(XTtheta) - 1) / norms_X_col


def norml2inf(A):
    return np.max(norm(A, ord=2, axis=1))


def norml21(A):
    return np.sum(norm(A, ord=2, axis=1))


def mt_primal(R, Beta, alpha):
    return (R ** 2).sum() / 2. + alpha * norml21(Beta)


def mt_dual(Y, Theta, alpha, norm_Y2):
    return norm_Y2 / 2. - alpha ** 2 / 2. * \
        ((Y / alpha - Theta) ** 2).sum()


def mt_feature_prios(Theta, X, norms_X_col):
    """Theta is feasible"""
    return (1. - norm(np.dot(X.T, Theta), axis=1)) / norms_X_col


def mt_compute_alpha(Theta, Ksi, X):
    # XtDiff = np.dot(X.T, Ksi - Theta)
    XtKsi = np.dot(X.T, Ksi)
    unmet = np.where(norm(XtKsi, axis=1) > 1.)[0]

    if len(unmet) == 0:
        return 1.

    XtTheta = np.dot(X.T[unmet, :], Theta)
    XtDiff = XtKsi[unmet, :] - XtTheta

    a = norm(XtDiff, axis=1, ord=2) ** 2
    b = 2 * np.sum(XtDiff * XtTheta, axis=1)
    c = norm(XtTheta, axis=1, ord=2) ** 2 - 1.
    delta = b ** 2 - 4. * a * c
    alphas = (- b + np.sqrt(delta)) / (2 * a)
    alphas = np.minimum(alphas, 1.)

    return np.min(alphas)


def configure_plt():
    import seaborn as sns
    from matplotlib import rc
    import matplotlib.pyplot as plt

    rc('font', **{'family': 'sans-serif', 'sans-serif': ['Computer Modern Roman']})
    params = {'axes.labelsize': 12,
              'font.size': 12,
              'legend.fontsize': 12,
              'xtick.labelsize': 10,
              'ytick.labelsize': 10,
              'text.usetex': True,
              'figure.figsize': (8, 6)}
    plt.rcParams.update(params)

    sns.set_context("poster")
    sns.set_style("ticks")


def plot_res(all_times, labels, tols, log=False,
             bottom=1e-1, savepath=None):
    n_competitors = len(all_times)
    fig, ax = plt.subplots(figsize=(7, 3.7))
    # if I do color='r' in ax.bar() it's not the red I want so I use this hack:
    prop_list = list(plt.rcParams['axes.prop_cycle'])
    colors = [prop_list[i]['color'] for i in range(n_competitors)]
    width = 0.2
    ind = np.arange(len(tols))
    if log:
        plt.yscale("log")
    for i in range(n_competitors):
        _ = ax.bar(ind + (i - 0.5) * width, all_times[i], width,
                   label=labels[i],
                   color=colors[i], bottom=bottom)
        # TODO: fix i - 0.5 to make it scale if there are 4, 5? 6 competitors
    # add some text for labels, title and axes ticks
    ax.set_ylabel('Time (s)')
    # ax.set_title('Comparison of Blitz and A5G on Leukemia')
    ax.set_xticks(ind + width / 2)
    from matplotlib.ticker import FormatStrFormatter
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.3e'))
    ax.set_xticklabels(["%.0e" % tol for tol in tols])

    ax.set_xlabel(r"$\bar{\epsilon}$")
    plt.legend(loc='best')
    plt.tight_layout()
    if savepath is not None:
        plt.savefig(savepath)
    plt.show()


def preprocess_data(X, y):
    y = y.copy()
    # add constant feature to fit intercept
    X = preprocessing.add_dummy_feature(X, 1.)
    # set all feature norms to 1
    X = preprocessing.normalize(X, axis=0)
    # center y
    y -= np.mean(y)
    # normalize y to get a first duality gap of 0.5
    y /= np.linalg.norm(y, ord=2)
    if sparse.issparse(X):
        X.sort_indices()

    return X, y
