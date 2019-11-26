import GPy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs

grid = gs.GridSpec(1,3,wspace = 0.05)
fig = plt.figure(figsize=(12,4), dpi =144)

np.random.seed(4)

ls = 1
va = 1
plot_h = 5

X = np.arange(0,8,0.01).reshape(-1,1)
mu = np.zeros(X.shape[0])

kern = GPy.kern.Linear(input_dim = 1)
C = kern.K(X,X)
sdv = np.sqrt(np.diag(C))
ax = fig.add_subplot(grid[0])

ax.text(0.1,1.05,'(a)', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize =14)
ax.fill_between(X.flatten(), (mu-sdv).flatten(), (mu+sdv).flatten(), facecolor = 'k', alpha = 0.3)
ax.fill_between(X.flatten(), (mu-2*sdv).flatten(), (mu+2*sdv).flatten(), facecolor = 'k', alpha = 0.2)
ax.fill_between(X.flatten(), (mu-3*sdv).flatten(), (mu+3*sdv).flatten(), facecolor = 'k', alpha = 0.1)
n = 5
samples = np.random.multivariate_normal(mu.flatten(), C, n)
for i in range(n):
    ax.plot(X[:], samples[i,:])
ax.set_xlim(xmin = 0, xmax = 8)
ax.set_ylim(ymin = -plot_h-7, ymax = plot_h+7)
ax.get_yaxis().set_visible(False)
ax.get_xaxis().set_visible(False)
ax.plot(X, mu, 'k')

kern = GPy.kern.RBF(input_dim = 1, variance = va, lengthscale = ls)
C = kern.K(X,X)
sdv = np.sqrt(np.diag(C))
ax = fig.add_subplot(grid[1])

ax.text(0.1,1.05,'(b)', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize =14)
ax.fill_between(X.flatten(), (mu-sdv).flatten(), (mu+sdv).flatten(), facecolor = 'k', alpha = 0.3)
ax.fill_between(X.flatten(), (mu-2*sdv).flatten(), (mu+2*sdv).flatten(), facecolor = 'k', alpha = 0.2)
ax.fill_between(X.flatten(), (mu-3*sdv).flatten(), (mu+3*sdv).flatten(), facecolor = 'k', alpha = 0.1)
n = 5
samples = np.random.multivariate_normal(mu.flatten(), C, n)
for i in range(n):
    ax.plot(X[:], samples[i,:])
ax.set_xlim(xmin = 0, xmax = 8)
ax.set_ylim(ymin = -plot_h, ymax = plot_h)
ax.get_yaxis().set_visible(False)
ax.get_xaxis().set_visible(False)
ax.plot(X, mu, 'k')


kern = GPy.kern.StdPeriodic(input_dim = 1, variance = va, period = 1.5, lengthscale = ls)
C = kern.K(X,X)
sdv = np.sqrt(np.diag(C))
ax = fig.add_subplot(grid[2])

ax.text(0.1,1.05,'(c)', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize =14)
ax.fill_between(X.flatten(), (mu-sdv).flatten(), (mu+sdv).flatten(), facecolor = 'k', alpha = 0.3)
ax.fill_between(X.flatten(), (mu-2*sdv).flatten(), (mu+2*sdv).flatten(), facecolor = 'k', alpha = 0.2)
ax.fill_between(X.flatten(), (mu-3*sdv).flatten(), (mu+3*sdv).flatten(), facecolor = 'k', alpha = 0.1)
n = 5
samples = np.random.multivariate_normal(mu.flatten(), C, n)
for i in range(n):
    ax.plot(X[:], samples[i,:])
ax.set_xlim(xmin = 0, xmax = 8)
ax.set_ylim(ymin = -plot_h, ymax = plot_h)
ax.get_yaxis().set_visible(False)
ax.get_xaxis().set_visible(False)
ax.plot(X, mu, 'k')


plt.show()
