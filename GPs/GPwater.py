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

def plot_gp(mu, cov, X,Y, X_train=None, Y_train=None, samples=[]):
    plt.plot(X,Y , 'k', alpha = 1)
#    plt.ylim(ymin = 0, ymax = 200)
    
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

mass = 1 #kg 
Ti = 21

LatentHeat_Fusion = 334 #kJ/kg
LatentHeat_Vapor = 2264.705 #kJ/kg
MeltingPoint = 0 #degC
BoilingPoint = 100.0 #degC
HeatCap_Ice = 2.108 #kJ/kg/C
HeatCap_Water = 4.148 #kJ/kg/C
HeatCap_Steam = 1.996 #kJ/kg/C

Energy = np.arange(0,500,1).reshape(-1,1)
Temp = []

Boiling_Energy = -1
for i in range(len(Energy)):
    T = Ti + ((1.0/HeatCap_Water) * Energy[i])
    if T > BoilingPoint:
        T = BoilingPoint
    Temp.append(T)

X_train = np.array([0,100,200,300,400]).reshape(-1,1)
Y_train = []
for E in enumerate(X_train):
    i = np.where(Energy == E[1])[0][0]
    Y_train.append(Temp[i])

mu_s, cov_s = posterior_predictive(Energy, X_train, Y_train, l =100)

#samples = np.random.multivariate_normal(mu_s.ravel(), cov_s, 3, tol=1e-6)

plot_gp(mu_s, cov_s, Energy, Temp, X_train=X_train, Y_train=Y_train)
