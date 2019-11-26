import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

np.random.seed(0)

X = np.array([1, 2, 3, 4, 5 , 6])
Y = X + np.random.randn(len(X))

m, c, r_value, p_value, std_err = stats.linregress(X,Y)

x = np.arange(0,8,0.01)
y = x*m + c

plt.plot(x,y, linewidth = 2.5, label = 'y = %.2f*x + %.2f' %(m,c))
plt.scatter(X,Y, c = 'k' , marker = 'D', zorder = 5)
for i in range(len(X)):
    y_slope = X[i]*m + c
    plt.plot([X[i],X[i]], [y_slope, Y[i]], linewidth = 2, c = 'r')

plt.xlim(xmin = 0, xmax = 8)
plt.ylim(ymin = 0, ymax = 8)
plt.legend()
plt.tight_layout()
plt.show()
