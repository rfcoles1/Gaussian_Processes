import matplotlib.pyplot as plt
import matplotlib.gridspec as gs 
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

seed = 100
np.random.seed(seed)
tf.set_random_seed(seed)

class network():
    def __init__(self):
        self.data = tf.placeholder(tf.float32, [None,1])
        self.target = tf.placeholder(tf.float32, [None,1])
        
        self.keep_prob = tf.placeholder(tf.float32)

        self.fc1 = slim.fully_connected(self.data, num_hidden,
            activation_fn = tf.nn.relu,
            weights_initializer = tf.contrib.layers.xavier_initializer(),
            biases_initializer = tf.zeros_initializer())
        self.drop1 = tf.nn.dropout(self.fc1, self.keep_prob)
        self.fc2 = slim.fully_connected(self.drop1, num_hidden,
            activation_fn = tf.nn.relu,
            weights_initializer = tf.contrib.layers.xavier_initializer(),
            biases_initializer = tf.zeros_initializer())
        self.drop2 = tf.nn.dropout(self.fc2, self.keep_prob)
        #self.fc3 = slim.fully_connected(self.drop2, num_hidden,
        #    activation_fn = tf.nn.relu,
        #    weights_initializer = tf.contrib.layers.xavier_initializer(),
        #    biases_initializer = tf.zeros_initializer())
        #self.drop3 = tf.nn.dropout(self.fc3, self.keep_prob)
                
        self.prediction = slim.fully_connected(self.drop2, 1, 
            activation_fn = None,
            weights_initializer = tf.contrib.layers.xavier_initializer(),
            biases_initializer = None)

        self.cost = tf.reduce_mean(tf.squared_difference(self.target, self.prediction))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(self.cost)

grid = gs.GridSpec(1,2, wspace = 0.05)
fig = plt.figure(figsize=(10,4),dpi=144)

def train(num_epoch):
    for epoch in range(num_epoch):
        for _ in range(num_batches):
            inp, out = x_train, y_train
            op, c = sess.run([net.optimizer, net.cost], {net.data: np.reshape(inp, [np.shape(inp)[0], 1]), \
                net.target: np.reshape(out, [np.shape(out)[0], 1]), \
                net.keep_prob: drop_rate})

def test():
    for i in range(len(x_test)):
        results = []
        for j in range(100):
            results.append(sess.run(net.prediction, {net.data: np.reshape(x_test[i], [1,1]),\
            net.keep_prob: drop_rate}))
        Mean[i] = np.mean(results)
        Std[i] = np.std(results)

def func1(x_vals, noise = 0):
    def indv(x):
        fnoise = np.random.uniform(-noise, noise)
        return(x**3  + fnoise)
    vec = [indv(x) for x in x_vals]
    return np.array(vec).flatten()

x_train = np.array([-1.6, -1,-0.4, .5, 1.2])
y_train = func1(x_train)
x_train = x_train.tolist()
y_train = y_train.tolist()

inc = 0.01
x_test = np.arange(-2,2,inc)
y_test = func1(x_test)

Mean = np.zeros(len(x_test))
Std = np.zeros(len(x_test))

num_hidden = 512
curr_epoch = 0
num_batches = 1
drop_rate = 0.5

net = network()
sess = tf.Session()
sess.run(tf.global_variables_initializer())
train(1000)
test()

ax = fig.add_subplot(grid[0])

ax.plot(x_test, y_test, c = 'k', label='True Function')
ax.plot(x_test, Mean, c = 'r', label='Predictions')
ax.fill_between(x_test, Mean - Std, Mean + Std, facecolor = 'k', alpha = .3)
ax.fill_between(x_test, Mean - 2*Std, Mean + 2*Std, facecolor = 'k', alpha = .2)
ax.fill_between(x_test, Mean - 3*Std, Mean + 3*Std, facecolor = 'k', alpha = .1)
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

x_train = np.array([-1.2,-0.6,-0.2,.2,0.6])
y_train = func1(x_train)
x_train = x_train.tolist()
y_train = y_train.tolist()

inc = 0.01
x_test = np.arange(-2,2,inc)
y_test = func1(x_test)

Mean = np.zeros(len(x_test))
Std = np.zeros(len(x_test))

num_hidden = 512
curr_epoch = 0
num_batches = 1
drop_rate = 0.5

num_nets = 10
nets = []
for i in range(num_nets):
    nets.append(network())

sess = tf.Session()
sess.run(tf.global_variables_initializer())
train(5000)
test()

ax = fig.add_subplot(grid[1])

ax.plot(x_test, y_test, c = 'k', label='True Function')
ax.plot(x_test, Mean, c = 'r', label='Predictions')
ax.fill_between(x_test, Mean - Std, Mean + Std, facecolor = 'k', alpha = .3)
ax.fill_between(x_test, Mean - 2*Std, Mean + 2*Std, facecolor = 'k', alpha = .2)
ax.fill_between(x_test, Mean - 3*Std, Mean + 3*Std, facecolor = 'k', alpha = .1)
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



