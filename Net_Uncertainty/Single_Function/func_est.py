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

def func(x_vals, noise = 0):
    def indv(x):
        fnoise = np.random.uniform(-noise, noise)
        return(np.sin(x * (2 * np.pi)) + fnoise)
    vec = [indv(x) + x for x in x_vals]
    return np.array(vec).flatten()

inc = 0.001
x_train = np.concatenate([np.arange(-2,-1.5,inc), np.arange(-1.0,0.5,inc)])
y_train = func(x_train)    

x_test = np.arange(-2,3,inc)
y_test = func(x_test)

x_train = x_train.tolist()
y_train = y_train.tolist()

num_hidden = 512
curr_epoch = 0
num_batches = 1
drop_rate = 0.5


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
    for i in range(len(x_test)):
        Preds[i] = sess.run(net.prediction, {net.data: np.reshape(x_test[i], [1,1]),\
            net.keep_prob: 1})
        results = []
        for j in range(100):
            results.append(sess.run(net.prediction, {net.data: np.reshape(x_test[i], [1,1]),\
            net.keep_prob: drop_rate}))
        Mean_DO[i] = np.mean(results)
        Std_DO[i] = np.std(results)

def add_data(x=1.75, width = 0.25):
    #newdata = np.arange(x-width, x+width, inc)
    #new_x = np.append(x_train, newdata)
    #new_y = np.append(y_train, func(newdata))
    currx = x - width
    while currx < x + width:
        x_train.append(currx)
        y_train.append(func([currx]))
        currx += inc
    #return new_x, new_y

def plt_discrep():
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

def plt_predict(filename = 'Prediction'):
    plt.clf()
    plt.plot(x_test, y_test, c = 'k', alpha = 0)
    plt.ylim(ymin = plt.ylim()[0], ymax = plt.ylim()[1])
    plt.plot(x_test,Preds, c = 'g')
    plt.scatter(x_test,Mean_DO, c = 'r', s = 6)
    plt.fill_between(x_test, Mean_DO - Std_DO, Mean_DO + Std_DO, facecolor = 'r', alpha = 0.5)
    plt.fill_between(x_test, Mean_DO - 2*Std_DO, Mean_DO + 2*Std_DO, facecolor = 'r', alpha = 0.25)
    plt.fill_between(x_test, Mean_DO - 3*Std_DO, Mean_DO + 3*Std_DO, facecolor = 'r', alpha = 0.1)
    plt.scatter(x_train, y_train, c = 'b', s = 12)   
    savefig(filename)

def plt_uncert(filename = 'Uncertainty'):
    plt.clf()
    plt.plot(x_test, Std_DO)
    plt.ylim(ymin = plt.ylim()[0])
    plt.plot([-1.5,-1.5], [-10,10], 'm--')
    plt.plot([-1.,-1.], [-10,10], 'm--')
    plt.plot([.5,.5], [-10,10], 'm--')
    savefig(filename)

def plt_mean(filename = 'Mean_Result'):
    plt.clf()
    plt.plot(x_test, y_test, c = 'k', alpha = 0)
    plt.ylim(ymin = plt.ylim()[0], ymax = plt.ylim()[1])
    plt.plot(x_test, Mean_DO, c = 'r')
    plt.plot(x_test,y_test, c = 'k')
    savefig(filename)

def good_plot(filename = 'File'):
    plt.clf()
    fig = plt.figure()
    gs = gridspec.GridSpec(2,1, height_ratios = [1,3])
    ax1 = plt.subplot(gs[0])
    plt.plot(x_test, Std_DO)
    plt.ylim(ymin = 0, ymax = 2)
    plt.ylabel('Abs Uncertainty')
    ax2 = plt.subplot(gs[1])
    plt.plot(x_test, y_test, c = 'k', alpha = 0)
    plt.ylim(ymin = plt.ylim()[0], ymax = plt.ylim()[1] + 2)
    #plt.plot(x_test, Preds, c = 'g')
    plt.scatter(x_test,Mean_DO, c = 'r', s = 6)
    plt.fill_between(x_test, Mean_DO - Std_DO, Mean_DO + Std_DO, facecolor = 'r', alpha = 0.5)
    plt.fill_between(x_test, Mean_DO - 2*Std_DO, Mean_DO + 2*Std_DO, facecolor = 'r', alpha = 0.25)
    plt.fill_between(x_test, Mean_DO - 3*Std_DO, Mean_DO + 3*Std_DO, facecolor = 'r', alpha = 0.1)
    plt.scatter(x_train, y_train, c = 'b', s = 12)
    plt.plot(x_test, y_test, c = 'k')
    savefig(filename)


def trainnewdata():
    train(10000)
    test()
    good_plot('base')
    add_data()
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
        moar_data(num)
        train(5000)
        num = test()
        i += 1
    #plt_predict(filename = 'Predict' + str(i))
    #plt_mean(filename = 'Mean' + str(i))
    good_plot(filename = 'Plot' + str(i))

