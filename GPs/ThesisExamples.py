import GPy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs

seed = 100
np.random.seed(seed)

grid = gs.GridSpec(1,2, wspace = 0.05)
fig = plt.figure(figsize=(10,4),dpi=144)

def func1(x_vals, noise = 0):
    def indv(x):
        fnoise = np.random.uniform(-noise, noise)
        return(x**3  + fnoise)
    vec = [indv(x) for x in x_vals]
    return np.array(vec).flatten()


x_train = np.array([-1.6, -1,-0.4, .5, 1.2]).reshape(-1,1)
y_train = func1(x_train).reshape(-1,1)
#x_train = x_train.tolist()
#y_train = y_train.tolist()

inc = 0.01
x_test = np.arange(-2,2,inc).reshape(-1,1)
y_test = func1(x_test).reshape(-1,1)

kern = GPy.kern.RBF(input_dim=1, variance=1.0, lengthscale=1.0)
model = GPy.models.GPRegression(x_train, y_train, kern)
noise = 0
model.Gaussian_noise.variance = noise**2
model.Gaussian_noise.variance.fix()

model.optimize()
[Mean, Std] = model.predict(x_test)
ax = fig.add_subplot(grid[0])

ax.plot(x_test, y_test, c = 'k', label='True Function')
ax.plot(x_test, Mean, c = 'r', label = 'Predictions')
ax.fill_between(x_test.flatten(), (Mean - Std).flatten(), (Mean + Std).flatten(), facecolor = 'k', alpha = .3)
ax.fill_between(x_test.flatten(), (Mean - 2*Std).flatten(), (Mean + 2*Std).flatten(), facecolor = 'k', alpha = .2)
ax.fill_between(x_test.flatten(), (Mean - 3*Std).flatten(), (Mean + 3*Std).flatten(), facecolor = 'k', alpha = .1)
#ax.set_xlabel('x')
#ax.set_ylabel('f(x)')
ax.set_xlim(left=-1.9,right=1.9)
ax.set_ylim(bottom=-8, top =8)
ax.scatter(x_train, y_train, c='k', marker = 'D', zorder = 10)
ax.text(0.05,1.05, '(a)', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
ax.legend()

def func1(x_vals, noise = 0):
    def indv(x):
        fnoise = np.random.uniform(-noise, noise)
        return(np.sin(x * (2 * np.pi)) + fnoise)
    vec = [indv(x) + x for x in x_vals]
    return np.array(vec).flatten()

x_train = np.array([-1.2,-0.6,-0.2,.2,0.6]).reshape(-1,1)
y_train = func1(x_train).reshape(-1,1)

inc = 0.01
x_test = np.arange(-2,2,inc).reshape(-1,1)
y_test = func1(x_test).reshape(-1,1)

kern = GPy.kern.RBF(input_dim=1, variance=1.0, lengthscale=0.3)
#kern1 = GPy.kern.StdPeriodic(input_dim = 1, variance=1.0)
#kern2 = GPy.kern.Linear(input_dim = 1)
#kern = GPy.kern.Add([kern1,kern2])
model = GPy.models.GPRegression(x_train, y_train, kern)
noise = 0
model.Gaussian_noise.variance = noise**2
model.Gaussian_noise.variance.fix()
model.kern.lengthscale.fix()
model.optimize()
[Mean, Std] = model.predict(x_test)
  
ax = fig.add_subplot(grid[1])

ax.plot(x_test, y_test, c = 'k', label= 'True Function')
ax.plot(x_test, Mean, c = 'r', label='Predictions')
ax.fill_between(x_test.flatten(), (Mean - Std).flatten(), (Mean + Std).flatten(), facecolor = 'k', alpha = .3)
ax.fill_between(x_test.flatten(), (Mean - 2*Std).flatten(), (Mean + 2*Std).flatten(), facecolor = 'k', alpha = .2)
ax.fill_between(x_test.flatten(), (Mean - 3*Std).flatten(), (Mean + 3*Std).flatten(), facecolor = 'k', alpha = .1)
ax.yaxis.tick_right()
#ax.set_xlabel('x')
#ax.set_ylabel('f(x)')
ax.set_xlim(left=-1.9,right=1.9)
ax.set_ylim(bottom=-3, top=3)
ax.scatter(x_train, y_train, c='k', marker = 'D', zorder = 10)
ax.text(0.05,1.05, '(b)', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
ax.legend()

plt.tight_layout()
plt.show()
                                            
