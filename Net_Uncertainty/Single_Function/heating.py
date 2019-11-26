import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import copy
from pylab import savefig
import tensorflow as tf
import tensorflow.contrib.slim as slim

class network():
    def __init__(self):
        self.data = tf.placeholder(tf.float32, [None,1])
        self.target = tf.placeholder(tf.float32, [None,1])

        self.keep_prob = tf.placeholder(tf.float32)

        self.fc1 = slim.fully_connected(self.data, num_hidden,
            activation_fn = tf.nn.relu,
            weights_initializer = tf.contrib.layers.xavier_initializer(),
            biases_initializer = tf.ones_initializer())
        self.drop1 = tf.nn.dropout(self.fc1, self.keep_prob)
        self.fc2 = slim.fully_connected(self.drop1, num_hidden,
            activation_fn = tf.nn.relu,
            weights_initializer = tf.contrib.layers.xavier_initializer(),
            biases_initializer = tf.zeros_initializer())
        self.drop2 = tf.nn.dropout(self.fc2, self.keep_prob)
        self.fc3 = slim.fully_connected(self.drop2, num_hidden,
            activation_fn = tf.nn.relu,
            weights_initializer = tf.contrib.layers.xavier_initializer(),
            biases_initializer = tf.zeros_initializer())
        self.drop3 = tf.nn.dropout(self.fc3, self.keep_prob)

        self.prediction = slim.fully_connected(self.drop3, 1,
            activation_fn = None,
            weights_initializer = tf.contrib.layers.xavier_initializer(),
            biases_initializer = None)

        self.cost = tf.reduce_mean(tf.squared_difference(self.target, self.prediction))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(self.cost)



mass = 1 #kg 
Ti = 21

LatentHeat_Fusion = 334 #kJ/kg
LatentHeat_Vapor = 2264.705 #kJ/kg
MeltingPoint = 0 #degC
BoilingPoint = 100.0 #degC
HeatCap_Ice = 2.108 #kJ/kg/C
HeatCap_Water = 4.148 #kJ/kg/C
HeatCap_Steam = 1.996 #kJ/kg/C

Energy = np.arange(0,500,1)

x_train = []
y_train = []

Boiling_Energy = -1
for i in range(len(Energy)):
    x_train.append(Energy[i])
    Temp = Ti + ((1.0/HeatCap_Water) * Energy[i])
    y_train.append(Temp)
    if Temp > BoilingPoint:
        Boiling_Energy = i
        break


#y_train = np.flip(y_train,0)
#y_train += 100
#y_train = y_train.tolist()


x_test = Energy
y_test = copy.deepcopy(y_train)

"""
for i in range(10):
    y_train.append(BoilingPoint)
    x_train.append(len(x_train))
"""

#while len(y_test) < 500: 
#    y_test.append(21)




for i in range(len(Energy)):
    if i > Boiling_Energy: #and i < Boiling_Energy + SLH:
        y_test.append(BoilingPoint)
    #if i > Boiling_Energy + SLH:
    #    y_test.append(BoilingPoint + (1.0/HeatCap_Steam) * (Energy[i] - (Boiling_Energy+SLH)))

#x_train = np.array(x_train) + (i-Boiling_Energy)
#y_train = np.flip(y_train,0)
#y_test = np.flip(y_test,0)

seed = 100
num_hidden = 512
curr_epoch = 0
num_batches = 1
drop_rate = 0.5

np.random.seed(seed)
tf.set_random_seed(seed)

net = network()
sess = tf.Session()
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
save_path = 'Dropout Uncertainty Test - 2 Layers, ' + str(num_hidden) + ' nodes, relu, ' + str(drop_rate) + ' drop rate ,'


Loss = []
def train(num_epoch):
    global curr_epoch
    for epoch in range(num_epoch):
        avg_cost = 0
        ptr = 0
        for _ in range(num_batches):
            inp, out = x_train, y_train
            op, c = sess.run([net.optimizer, net.cost], {net.data: np.reshape(inp, [np.shape(inp)[0], 1]), \
                net.target: np.reshape(out, [np.shape(out)[0], 1]), \
                net.keep_prob: drop_rate})
            avg_cost += c
        avg_cost/=num_batches
        Loss.append(avg_cost)
        if curr_epoch%100 == 0 and curr_epoch!= 0:
            print curr_epoch, avg_cost
            if curr_epoch%1000 == 0:
                saver.save(sess, './models/' + save_path + ' epoch ' + str(epoch))
        curr_epoch += 1

def plt_train(filename = 'Training_Data'):
    plt.clf()
    plt.plot(x_test, y_test, c = 'k', alpha = 0)
    plt.scatter(x_train, y_train, c = 'b', s = 12)
    savefig(filename)

def plt_test(filename = 'Testing_Data'):
    plt.clf()
    plt.plot(x_test, y_test, c = 'k')
    savefig(filename)

def plt_loss(filename = 'Loss'):
    plt.clf()
    plt.plot(Loss)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.yscale('log')
    savefig(filename)

Preds = np.zeros(len(x_test))
Mean_DO = np.zeros(len(x_test))
Std_DO = np.zeros(len(x_test))


def test():
    #test_num = -1
    for i in range(len(x_test)):
        Preds[i] = sess.run(net.prediction, {net.data: np.reshape(x_test[i], [1,1]),\
            net.keep_prob: 1})
        results = []
        for j in range(250):
            results.append(sess.run(net.prediction, {net.data: np.reshape(x_test[i], [1,1]),\
            net.keep_prob: drop_rate}))
        Mean_DO[i] = np.mean(results)
        Std_DO[i] = np.std(results)
        #if Std_DO[i] > 8 and test_num == -1:
        #    test_num = x_test[i]
        #    print test_num
    #return test_num

def add_data(i,width = 10):
    for i in range(i-width, i+width):
        x_train.append(x_test[i])
        y_train.append(y_test[i])
    

def plt_discrep(filename = 'dropout_discrepency'):
    plt.clf()
    plt.scatter(Mean_DO, Preds)
    axes = plt.gca()
    ymin, ymax = axes.get_ylim()
    xmin, xmax = axes.get_xlim()
    axes.set_ylim(min(ymin, xmin), max(ymax, xmax))
    axes.set_xlim(min(ymin, xmin), max(ymax, xmax))
    
    plt.xlabel('Average prediction with Dropout')
    plt.ylabel('Prediction without Dropout')
    savefig(filename)

def plt_predict(num = -1, filename = 'Prediction'):
    #plt.ylim(ymin = plt.ylim()[0], ymax = plt.ylim()[1])
    plt.clf()
    plt.ylim(0,200)
    plt.scatter(x_test,Mean_DO, c = 'r', s = 6)
    plt.fill_between(x_test, Mean_DO - Std_DO, Mean_DO + Std_DO, facecolor = 'r', alpha = 0.5)
    plt.fill_between(x_test, Mean_DO - 2*Std_DO, Mean_DO + 2*Std_DO, facecolor = 'r', alpha = 0.25)
    plt.fill_between(x_test, Mean_DO - 3*Std_DO, Mean_DO + 3*Std_DO, facecolor = 'r', alpha = 0.1)
    plt.scatter(x_train, y_train, c = 'b', s = 12)
    plt.plot(x_test, y_test, c = 'k')
    plt.xlabel('Energy Input (KJ)')
    plt.ylabel('Temperature ($^\circ$C)')
    
    if num != -1:
        plt.plot([x_test[num], x_test[num]], [0,200], 'c--')
    savefig(filename)

def plt_uncert(filename = 'Uncertainty'):
    plt.clf()
    plt.plot(x_test, Std_DO)
    plt.ylim(ymin = plt.ylim()[0])
    plt.plot([Boiling_Energy, Boiling_Energy], [0,100], 'm--')
    savefig(filename)

def plt_mean(filename = 'Mean_Result'):
    plt.clf()
    plt.ylim(0,200)
    plt.plot(x_test, Mean_DO, c = 'r')
    plt.plot(x_test,y_test, c = 'k')
    savefig(filename)

def good_plot(filename = 'File'):
    plt.clf()
    fig = plt.figure()
    gs = gridspec.GridSpec(3,1, height_ratios = [1,1,3])
    ax1 = plt.subplot(gs[0])
    plt.plot(x_test, Std_DO/Mean_DO)
    plt.ylabel('Rel Uncertainty')
    plt.ylim(0,0.15)
    ax2 = plt.subplot(gs[1])
    plt.plot(x_test, Std_DO)
    plt.ylabel('Abs Uncertainty')
    plt.ylim(0,15)
    ax3 = plt.subplot(gs[2])
    plt.ylim(0,200)
    plt.scatter(x_test,Mean_DO, c = 'r', s = 6)
    plt.fill_between(x_test, Mean_DO - Std_DO, Mean_DO + Std_DO, facecolor = 'r', alpha = 0.5)
    plt.fill_between(x_test, Mean_DO - 2*Std_DO, Mean_DO + 2*Std_DO, facecolor = 'r', alpha = 0.25)
    plt.fill_between(x_test, Mean_DO - 3*Std_DO, Mean_DO + 3*Std_DO, facecolor = 'r', alpha = 0.1)
    plt.scatter(x_train, y_train, c = 'b', s = 12)
    plt.plot(x_test, y_test, c = 'k')
    plt.xlabel('Energy Input (KJ)')
    plt.ylabel('Temperature ($^\circ$C)')
    savefig(filename)

def trainnewdata():
    train(10000)
    test()
    good_plot('base')
    add_data(425)
    epochs = [1000,1500,2500,5000,10000]
    curr = 0
    for x in epochs:
        train(x)
        curr += x
        test()
        good_plot(str(curr))


def makeitgo():
    train(5000)
    num = test()
    i = 1
    while num != -1:
        #plt_predict(filename = 'Predict' + str(i))
        #plt_mean(filename = 'Mean' + str(i))
        good_plot(filename = 'Plot' + str(i))
        add_data(num)
        train(5000)
        num = test()
        i += 1
    #plt_predict(filename = 'Predict' + str(i))
    #plt_mean(filename = 'Mean' + str(i))
    good_plot(filename = 'Plot' + str(i))
