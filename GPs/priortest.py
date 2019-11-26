import GPy 
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

ls = 1
va = 1
plot_h = 5

X = np.arange(0,8,0.01).reshape(-1,1)
kern = GPy.kern.RBF(input_dim = 1, variance = va, lengthscale = ls)
#kern = GPy.kern.RatQuad(input_dim = 1, variance = 1.0, lengthscale = 1.0)
#kern = GPy.kern.PeriodicMatern32(input_dim = 1, variance = 1.0, lengthscale = 1.0)


#kern1 = GPy.kern.Linear(input_dim = 1)
#kern2 = GPy.kern.RBF(input_dim = 1, variance = 1.0, lengthscale = 1.0)
#kern = GPy.kern.Add([kern1, kern2])

mu = np.zeros(X.shape[0]) 
def func(x):
    return 0

"""
mu = np.multiply(np.ones(X.shape[0]),X.T)
def func(x):
    return x
mu = np.sin(X)
def func(x):
    return np.sin(x)
"""
mf = GPy.core.Mapping(1,1)
mf.f = func
mf.update_gradients = lambda a,b: None

C = kern.K(X,X) #covariance matrix
sdv = np.sqrt(np.diag(C))

#first observe the prior
plt.figure(1)

plt.fill_between(X.flatten(), (mu-sdv).flatten(), (mu+sdv).flatten(), facecolor = 'k', alpha = 0.3)
plt.fill_between(X.flatten(), (mu-2*sdv).flatten(), (mu+2*sdv).flatten(), facecolor = 'k', alpha = 0.2)
plt.fill_between(X.flatten(), (mu-3*sdv).flatten(), (mu+3*sdv).flatten(), facecolor = 'k', alpha = 0.1)
n = 5
samples = np.random.multivariate_normal(mu.flatten(), C, n)
for i in range(n):
    plt.plot(X[:], samples[i,:])
plt.xlim(xmin = 0, xmax = 8)
plt.ylim(ymin = -plot_h, ymax = plot_h)
plt.title('Prior - LinRBF(lengthscale = %.2f, variance = %.2f)' % (ls, va))
plt.plot(X, mu, 'k')
plt.tight_layout()

#data set
input_memory = np.array([0, 1, 1.25, 3,3.1, 5 ,5.55,5.6]).reshape(-1,1)
output_memory = np.array([-.1, 0.9, 1, 0.5, 0.45, -.1,0,.1]).reshape(-1,1)

#obtan the posterior
model = GPy.models.GPRegression(np.array(input_memory).reshape(-1,1),\
    np.array(output_memory).reshape(-1,1), kern, mean_function = mf)

noise = 0
model.Gaussian_noise.variance = noise**2
model.Gaussian_noise.variance.fix()
#model.kern.variance.fix()
#model.kern.lengthscale.fix()
model.optimize()

plt.figure(2)
n = 5
samples = model.posterior_samples_f(X, n)
samples = samples.reshape(-1, n).T

model.optimize()
[mean, sdv] = model.predict(X)
#ls = model.kern.lengthscale.values[0]
#va = model.kern.variance.values[0]

plt.fill_between(X.flatten(), (mean-sdv).flatten(), (mean+sdv).flatten(), facecolor = 'k', alpha = 0.3)
plt.fill_between(X.flatten(), (mean-2*sdv).flatten(), (mean+2*sdv).flatten(), facecolor = 'k', alpha = 0.2)
plt.fill_between(X.flatten(), (mean-3*sdv).flatten(), (mean+3*sdv).flatten(), facecolor = 'k', alpha = 0.1)
for i, sample in enumerate(samples):
    plt.plot(X, sample)
plt.scatter(input_memory, output_memory, c='k', marker = 'D', zorder = 10)
plt.plot(X, mean, 'k')
plt.xlim(xmin = 0, xmax = 8)
plt.ylim(ymin = -plot_h, ymax = plot_h)
plt.title('Posterior - LinRBF(lengthscale = %.2f, variance = %.2f)' % (ls, va))
plt.tight_layout()
plt.show()

