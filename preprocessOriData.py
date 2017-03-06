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
    column_names = ['user-id', 'activity', 'timestamp', 'x-axis', 'y-axis', 'z-axis']
    data = pd.read_csv(file_path, header=None, names=column_names, low_memory=False)
    # print(data[0])
    return data


def feature_normalize(dataset):
    mu = np.mean(dataset, axis=0)
    sigma = np.std(dataset, axis=0)
    return (dataset - mu) / sigma


def plot_axis(ax, x, y, title):
    ax.plot(x, y)
    ax.set_title(title)
    ax.xaxis.set_visible(False)
    ax.set_ylim([min(y) - np.std(y), max(y) + np.std(y)])
    ax.set_xlim([min(x), max(x)])
    ax.grid(True)


def plot_activity(activity, data):
    fig, (ax0, ax1, ax2) = plt.subplots(nrows=3, figsize=(15, 10), sharex=True)
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


def segment_signal(data, window_size=90):
    segments = np.empty((0, window_size, 3))
    labels = np.empty((0))
    for (start, end) in windows(data['timestamp'], window_size):
        x = data["x-axis"][start:end]
        y = data["y-axis"][start:end]
        z = data["z-axis"][start:end]
        if (len(dataset['timestamp'][start:end]) == window_size):
            segments = np.vstack([segments, np.dstack([x, y, z])])
            labels = np.append(labels, stats.mode(data["activity"][start:end])[0][0])
    return segments, labels


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.0, shape=shape)
    return tf.Variable(initial)


def depthwise_conv2d(x, W):
    return tf.nn.depthwise_conv2d(x, W, [1, 1, 1, 1], padding='VALID')


def apply_depthwise_conv(x, kernel_size, num_channels, depth):
    weights = weight_variable([1, kernel_size, num_channels, depth])
    biases = bias_variable([depth * num_channels])
    return tf.nn.relu(tf.add(depthwise_conv2d(x, weights), biases))


def apply_max_pool(x, kernel_size, stride_size):
    return tf.nn.max_pool(x, ksize=[1, 1, kernel_size, 1],
                          strides=[1, 1, stride_size, 1], padding='VALID')


# In[11]:

dataset = read_data('./actitracker_new.txt')
# dataset = read_data('test.txt')
print("get data")
dataset['x-axis'] = feature_normalize(dataset['x-axis'])
dataset['y-axis'] = feature_normalize(dataset['y-axis'])
# z_axis = dataset['z-axis']
dataset['z-axis'] = feature_normalize(dataset['z-axis'])
print("normalized")

# In[ ]:

# for activity in np.unique(dataset["activity"]):
#     subset = dataset[dataset["activity"] == activity][:180]
#     plot_activity(activity,subset)


# In[ ]:

segments, labels = segment_signal(dataset)
# segmentsData = open('segmentData.pkl', 'wb')
# pickle.dump(segments, segmentsData)
# labelsData = open('labelsData.pkl', 'wb')
# pickle.dump(labels, labelsData)

labels = np.asarray(pd.get_dummies(labels), dtype=np.int8)
reshaped_segments = segments.reshape(len(segments), 1, 90, 3)

# In[ ]:

train_test_split = np.random.rand(len(reshaped_segments)) < 0.70
train_x = reshaped_segments[train_test_split]
train_y = labels[train_test_split]
test_x = reshaped_segments[~train_test_split]
test_y = labels[~train_test_split]

data = [train_x, train_y, test_x, test_y]
processedData = open('./processedData/processedOriData.pkl', 'wb')
pickle.dump(data, processedData)