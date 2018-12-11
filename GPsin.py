import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt

def kernel(X1, X2, l=1.0, sigma_f=1.0):
    sqdist = np.sum(X1**2, 1).reshape(-1,1) + np.sum(X2**2, 1) - 2 * np.dot(X1, X2.T)
    return sigma_f**2 * np.exp(-0.5 / l**2 * sqdist)

def posterior_predictive(X_s, X_train, Y_train, l=1.0, sigma_f=1.0, sigma_y=1e-8):
    K = kernel(X_train, X_train, l, sigma_f) + sigma_y**2 * np.eye(len(X_train))
    K_s = kernel(X_train, X_s, l, sigma_f)
    K_ss = kernel(X_s, X_s, l, sigma_f) + 1e-8*np.eye(len(X_s))
    K_inv = inv(K)

    mu_s = K_s.T.dot(K_inv).dot(Y_train)
    cov_s = K_ss - K_s.T.dot(K_inv).dot(K_s)
    return mu_s, cov_s

def plot_gp(mu, cov, X, X_train=None, Y_train=None, samples=[]):
    plt.plot(X, np.sin(X * 2*np.pi) + X, 'k', alpha = 1)
    plt.ylim(ymin = plt.ylim()[0], ymax = plt.ylim()[1] + 2)
    plt.xlim(xmin = -2, xmax = 8)
    
    
    X = X.ravel()
    mu = mu.ravel()
    uncertainty = 1.96*np.sqrt(np.diag(cov))

    plt.fill_between(X, mu + uncertainty, mu - uncertainty, alpha = 0.1)
    plt.plot(X, mu)
    for i, sample in enumerate(samples):
        plt.plot(X, sample, ls='--')
    if X_train is not None:
        plt.plot(X_train, Y_train, 'rx')
    plt.legend()
    plt.show()

X = np.arange(-2,8,0.01).reshape(-1,1)

X_train = np.array([-1,-.26,0.32,1.2,2.11,3,5]).reshape(-1,1)
Y_train = np.sin(X_train * 2*np.pi) + X_train

mu_s, cov_s = posterior_predictive(X, X_train, Y_train)

samples = np.random.multivariate_normal(mu_s.ravel(), cov_s, 3, tol=1e-6)

plot_gp(mu_s, cov_s, X, X_train=X_train, Y_train=Y_train)
