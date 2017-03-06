
# coding: utf-8

# In[9]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import tensorflow as tf
import pickle

# get_ipython().magic(u'matplotlib inline')
# %matplotlib inline
plt.style.use('ggplot')


# In[10]:

def read_data(file_path):
    column_names = ['user-id','activity','timestamp', 'x-axis', 'y-axis', 'z-axis']
    data = pd.read_csv(file_path,header = None, names = column_names, low_memory=False)
    # print(data[0])
    return data

def feature_normalize(dataset):
    mu = np.mean(dataset,axis = 0)
    sigma = np.std(dataset,axis = 0)
    return (dataset - mu)/sigma
    
def plot_axis(ax, x, y, title):
    ax.plot(x, y)
    ax.set_title(title)
    ax.xaxis.set_visible(False)
    ax.set_ylim([min(y) - np.std(y), max(y) + np.std(y)])
    ax.set_xlim([min(x), max(x)])
    ax.grid(True)
    
def plot_activity(activity,data):
    fig, (ax0, ax1, ax2) = plt.subplots(nrows = 3, figsize = (15, 10), sharex = True)
    plot_axis(ax0, data['timestamp'], data['x-axis'], 'x-axis')
    plot_axis(ax1, data['timestamp'], data['y-axis'], 'y-axis')
    plot_axis(ax2, data['timestamp'], data['z-axis'], 'z-axis')
    plt.subplots_adjust(hspace=0.2)
    fig.suptitle(activity)
    plt.subplots_adjust(top=0.90)
    plt.show()
    
def windows(data, size):
    start = 0
    while start < data.count():
        yield start, start + size
        start += (size / 2)

def segment_signal(data,window_size = 90):
    segments = np.empty((0,window_size,3))
    labels = np.empty((0))
    for (start, end) in windows(data['timestamp'], window_size):
        x = data["x-axis"][start:end]
        y = data["y-axis"][start:end]
        z = data["z-axis"][start:end]
        if(len(dataset['timestamp'][start:end]) == window_size):
            segments = np.vstack([segments,np.dstack([x,y,z])])
            labels = np.append(labels,stats.mode(data["activity"][start:end])[0][0])
    return segments, labels

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev = 0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.0, shape = shape)
    return tf.Variable(initial)

def depthwise_conv2d(x, W):
    return tf.nn.depthwise_conv2d(x,W, [1, 1, 1, 1], padding='VALID')

def apply_depthwise_conv(x,kernel_size,num_channels,depth):
    weights = weight_variable([1, kernel_size, num_channels, depth])
    biases = bias_variable([depth * num_channels])
    return tf.nn.relu(tf.add(depthwise_conv2d(x, weights),biases))
    
def apply_max_pool(x,kernel_size,stride_size):
    return tf.nn.max_pool(x, ksize=[1, 1, kernel_size, 1], 
                          strides=[1, 1, stride_size, 1], padding='VALID')


# In[11]:
#
# dataset = read_data('../actitracker_new9.txt')
# # dataset = read_data('test.txt')
# print("get data")
# dataset['x-axis'] = feature_normalize(dataset['x-axis'])
# dataset['y-axis'] = feature_normalize(dataset['y-axis'])
# # z_axis = dataset['z-axis']
# dataset['z-axis'] = feature_normalize(dataset['z-axis'])
# print("normalized")
#
# # In[ ]:
#
# # for activity in np.unique(dataset["activity"]):
# #     subset = dataset[dataset["activity"] == activity][:180]
# #     plot_activity(activity,subset)
#
#
# # In[ ]:
#
# segments, labels = segment_signal(dataset)
# # segmentsData = open('segmentData.pkl', 'wb')
# # pickle.dump(segments, segmentsData)
# # labelsData = open('labelsData.pkl', 'wb')
# # pickle.dump(labels, labelsData)
#
# labels = np.asarray(pd.get_dummies(labels), dtype = np.int8)
# reshaped_segments = segments.reshape(len(segments), 1,90, 3)
#
#
# # In[ ]:
#
# train_test_split = np.random.rand(len(reshaped_segments)) < 0.70
# train_x = reshaped_segments[train_test_split]
# train_y = labels[train_test_split]
# test_x = reshaped_segments[~train_test_split]
# test_y = labels[~train_test_split]
#
# data=[train_x, train_y, test_x, test_y]
# processedData = open('processedData.pkl', 'wb')
# pickle.dump(data, processedData)

processedData = open('../processedData/processedData.pkl', 'rb')
processedData = pickle.load(processedData)
# train_x = processedData[0]
train_x, train_y, test_x, test_y = processedData[0], processedData[1], processedData[2], processedData[3]
# In[12]:

input_height = 1
input_width = 90
num_labels = 6
num_channels = 3

batch_size = 10
kernel_size = 60
depth = 60
num_hidden = 1000

learning_rate = 0.0001
training_epochs = 100
lstm_size = 128

total_batches = train_x.shape[0] // batch_size

n_classes = 6
n_hidden = 128
n_inputs = 180

rnnW = {
    'hidden': tf.Variable(tf.random_normal([n_inputs, n_hidden])),
    'output': tf.Variable(tf.random_normal([n_hidden, n_classes]))
}
rnnBiases = {
    'hidden': tf.Variable(tf.random_normal([n_hidden], mean=1.0)),
    'output': tf.Variable(tf.random_normal([n_classes]))
}
# In[14]:

X = tf.placeholder(tf.float32, shape=[None,input_height,input_width,num_channels])
Y = tf.placeholder(tf.float32, shape=[None,num_labels])

c = apply_depthwise_conv(X,kernel_size,num_channels,depth)
p = apply_max_pool(c,20,2)
c2 = apply_depthwise_conv(p,6,depth*num_channels,depth//10)
c2Reshape = tf.reshape(c2, [-1, 6, 180])
shuff = tf.transpose(c2Reshape, [1, 0, 2])
shuff = tf.reshape(shuff, [-1, n_inputs])

# Linear activation, reshaping inputs to the LSTM's number of hidden:
hidden = tf.nn.relu(tf.matmul(
    shuff, rnnW['hidden']
) + rnnBiases['hidden'])
# X_split = tf.split(shuff, 8, 0) # split them to time_step_size (28 arrays)
# X_split = tf.unstack(shuff) # split them to time_step_size (28 arrays)

# Split the series because the rnn cell needs time_steps features, each of shape:
hidden = tf.split(axis=0, num_or_size_splits=6, value=hidden)

lstm_cell = tf.contrib.rnn.LSTMCell(lstm_size, forget_bias=1.0)
# Stack two LSTM layers, both layers has the same shape
lstm_layers = tf.contrib.rnn.MultiRNNCell([lstm_cell] * 2)

lstmOutputs, _ = tf.contrib.rnn.static_rnn(lstm_layers, hidden, dtype=tf.float32)
lstmLastOutput = lstmOutputs[-1]
y_ = tf.matmul(lstmLastOutput, rnnW['output']) + rnnBiases['output']

# X_split = tf.split(shuff, 8, 0) # split them to time_step_size (28 arrays)
# X_split = tf.unstack(shuff)  # split them to time_step_size (28 arrays)
# lstm_cell = tf.contrib.rnn.BasicLSTMCell(lstm_size, forget_bias=0.2, state_is_tuple=True)
# # Stack two LSTM layers, both layers has the same shape
# lstm_layers = tf.contrib.rnn.MultiRNNCell([lstm_cell] * 2)
#
# lstmOutputs, _ = tf.contrib.rnn.static_rnn(lstm_layers, X_split, dtype=tf.float32)
# lstmOutputsStack = tf.stack(lstmOutputs)
# lstm2denseShp = tf.transpose(lstmOutputsStack, [1, 0, 2])
#
#
# shape = lstm2denseShp.get_shape().as_list()
# c_flat = tf.reshape(lstm2denseShp, [-1, shape[1] * shape[2]])
#
# f_weights_l1 = weight_variable([shape[1] * shape[2], num_hidden])
# f_biases_l1 = bias_variable([num_hidden])
# f = tf.nn.tanh(tf.add(tf.matmul(c_flat, f_weights_l1),f_biases_l1))
#
# out_weights = weight_variable([num_hidden, num_labels])
# out_biases = bias_variable([num_labels])
# y_ = tf.nn.softmax(tf.matmul(f, out_weights) + out_biases)

with tf.name_scope("cost"):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_, labels=Y))
    # optimizer = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(loss)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)

    # Add scalar summary for cost
    tf.summary.scalar("loss", loss)

with tf.name_scope("accuracy"):
    correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(y_, 1)) # Count correct predictions
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float")) # Cast boolean to float to average
    # Add scalar summary for accuracy
    tf.summary.scalar("accuracy", accuracy)
# In[ ]:
# loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_, labels=Y))
# # loss = -tf.reduce_sum(Y * tf.log(y_))
# optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate).minimize(loss)
#
# correct_prediction = tf.equal(tf.argmax(y_,1), tf.argmax(Y,1))
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
session_conf = tf.ConfigProto()
session_conf.gpu_options.allow_growth = True
# session_conf.gpu_options.allocator_type = "BFC"
print("net work done")

# In[ ]:

cost_history = np.empty(shape=[1],dtype=float)

with tf.Session(config=session_conf) as session:
    # create a log writer. run 'tensorboard --logdir=./logs/nn_logs'
    writer = tf.summary.FileWriter("tfVis/logs/nn_logs", session.graph)  # for 1.0
    merged = tf.summary.merge_all()
    # you need to initialize all variables
    tf.global_variables_initializer().run()
    for epoch in range(training_epochs):
        for b in range(total_batches):    
            offset = (b * batch_size) % (train_y.shape[0] - batch_size)
            batch_x = train_x[offset:(offset + batch_size), :, :, :]
            batch_y = train_y[offset:(offset + batch_size), :]
            _, c = session.run([optimizer, loss],feed_dict={X: batch_x, Y : batch_y})
            cost_history = np.append(cost_history,c)

            test_indices = np.arange(len(test_x))  # Get A Test Batch
            np.random.shuffle(test_indices)
            test_indices = test_indices[0: 1000]
            summary, acc=session.run([merged, accuracy], feed_dict={X: test_x[test_indices], Y: test_y[test_indices]})
        print "Epoch: ",epoch," Training Loss: ",c," Training Accuracy: ", acc
        writer.add_summary(summary, epoch)  # Write summary

    print "Testing Accuracy:", session.run(accuracy, feed_dict={X: test_x, Y: test_y})
    writer.close()
