import GPy
import numpy as np
import matplotlib.pyplot as plt

noise = 0.0
X = np.arange(-2,8,0.01).reshape(-1,1)

mass = 1 #kg 
Ti = 21

LatentHeat_Fusion = 334 #kJ/kg
LatentHeat_Vapor = 2264.705 #kJ/kg
MeltingPoint = 0 #degC
BoilingPoint = 100.0 #degC
HeatCap_Ice = 2.108 #kJ/kg/C
HeatCap_Water = 4.148 #kJ/kg/C
HeatCap_Steam = 1.996 #kJ/kg/C

Energy = np.arange(0,600,1).reshape(-1,1)
Temp = []

Boiling_Energy = -1
for i in range(len(Energy)):
    T = Ti + ((1.0/HeatCap_Water) * Energy[i])
    if T > BoilingPoint:
        T = BoilingPoint
    Temp.append(T)

X_train = np.array([0,10]).reshape(-1,1)
Y_train = np.zeros_like(X_train)
for j in range(len(X_train)):
    E = X_train[j]
    i = np.where(Energy == E[0])[0][0]
    Y_train[j] = Temp[i]

Y_train = np.array(Y_train).reshape(-1,1)

kern = GPy.kern.RBF(input_dim=1, variance=.1, lengthscale=170.0)
model = GPy.models.GPRegression(X_train, Y_train, kern)


model.Gaussian_noise.variance = noise**2
model.Gaussian_noise.variance.fix()
model.kern.lengthscale.fix()
model.optimize()

l = model.kern.lengthscale.values[0]
sigma_f = np.sqrt(model.kern.variance.values[0])

Means = np.zeros(len(Energy))
Sdvs = np.zeros(len(Energy))
for i in range(len(Energy)):
    out =  model._raw_predict(np.array([Energy[i]]).reshape(-1,1))
    Means[i] = out[0].flatten()[0]
    Sdvs[i] = out[1].flatten()[0]

plt.plot(Energy, Temp, 'k', label = 'True')
plt.plot(Energy, Means, label = 'Mean')
plt.fill_between(Energy.flatten(), Means - Sdvs, Means + Sdvs, facecolor = 'r', alpha = 0.5, label = '1 sigma range')
plt.xlim(xmin = 0, xmax = 600)
plt.ylim(ymin = 0, ymax = 120)
plt.ylabel('Temperature ($^\circ$C)')
plt.xlabel('Energy Added (kJ)')
plt.tight_layout()
plt.legend()
plt.show()
