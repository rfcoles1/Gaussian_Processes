import GPy
import numpy as np
import matplotlib.pyplot as plt

noise = 0
X = np.arange(-2,8,0.01).reshape(-1,1)

X_train = np.array([-1,-.26,.32,1.2,2.11,3,5]).reshape(-1,1)
Y_train = np.sin(X_train * 2*np.pi) + X_train

rbf = GPy.kern.RBF(input_dim=1, variance=1.0, lengthscale=1.0)
gpr = GPy.models.GPRegression(X_train, Y_train, rbf)

gpr.Gaussian_noise.variance = noise**2
gpr.Gaussian_noise.variance.fix()

gpr.optimize()

l = gpr.rbf.lengthscale.values[0]
sigma_f = np.sqrt(gpr.rbf.variance.values[0])

fig = gpr.plot()
plt.show()
