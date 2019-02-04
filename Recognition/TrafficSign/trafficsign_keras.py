#!/usr/bin/env python

##############################################
### Download Data
##############################################
from urllib.request import urlretrieve
from os.path import isfile
from tqdm import tqdm

class DLProgress(tqdm):
    last_block = 0

    def hook(self, block_num=1, block_size=1, total_size=None):
        self.total = total_size
        self.update((block_num - self.last_block) * block_size)
        self.last_block = block_num

if not isfile('train.p'):
    with DLProgress(unit='B', unit_scale=True, miniters=1, desc='Train Dataset') as pbar:
        urlretrieve(
             'https://s3.amazonaws.com/udacity-sdc/datasets/german_traffic_sign_benchmark/train.p',
             'train.p',
             pbar.hook)

if not isfile('test.p'):
    with DLProgress(unit='B', unit_scale=True, miniters=1, desc='Test Dataset') as pbar:
        urlretrieve(
             'https://s3.amazonaws.com/udacity-sdc/datasets/german_traffic_sign_benchmark/test.p',
             'test.p',
             pbar.hook)

print('Training and Test data downloaded.')

##############################################
### Load Data
##############################################
import pickle
import numpy as np
import math

import tensorflow as tf

with open('train.p', 'rb') as f:
    data = pickle.load(f)

x_train = data['features']
y_train = data['labels']
assert(np.array_equal(x_train, data['features']))
assert(np.array_equal(y_train, data['labels']))
print('Tests passed.')

##############################################
### Preprocess Data
##############################################
from sklearn.utils import shuffle
x_train, y_train = shuffle(x_train, y_train)

# Normalize the features
def normalize_grayscale(image_data):
    a = -0.5
    b = 0.5
    grayscale_min = 0
    grayscale_max = 255
    return a + (((image_data - grayscale_min)*(b - a))/(grayscale_max - grayscale_min))

x_normalized = normalize_grayscale(x_train)

assert(math.isclose(np.min(x_normalized), -0.5, abs_tol=1e-5) and math.isclose(np.max(x_normalized), 0.5, abs_tol=1e-5))
print('Tests passed.')

# One-hot Encode the labels
from sklearn.preprocessing import LabelBinarizer
label_binarizer = LabelBinarizer()
y_one_hot = label_binarizer.fit_transform(y_train)

assert(y_one_hot.shape == (39209, 43))
print('Tests passed.')

##############################################
### Keras Sequential Model
##############################################
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Conv2D
from keras.activations import relu, softmax
from keras.layers.pooling import MaxPooling2D

# Create the Sequential model
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(32, 32, 3)))
model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.5))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dense(43))
model.add(Activation('softmax'))

# Test of keras model
def check_layers(layers, true_layers):
    assert(len(true_layers) != 0)
    for layer_i in range(len(layers)):
        assert(isinstance(true_layers[layer_i], layers[layer_i]))
    assert(len(true_layers) == len(layers))

# check_layers([Conv2D, MaxPooling2D, Dropout, Activation, Flatten, Dense, Activation, Dense, Activation], model.layers)
assert(model.layers[0].input_shape == (None, 32, 32, 3))
assert(model.layers[0].filters == 32)
assert(model.layers[0].kernel_size[0] == model.layers[0].kernel_size[1] == 3)
assert(model.layers[0].padding == 'valid')
print('Model Tests Passed.')


##############################################
### Training the Network
##############################################
model.compile('adam', 'categorical_crossentropy', ['accuracy'])
history = model.fit(x_normalized, y_one_hot, batch_size=128, epochs=10, validation_split=0.2, verbose=2)

from keras.optimizers import Adam

assert(model.loss == 'categorical_crossentropy')
assert(isinstance(model.optimizer, Adam))
assert(len(history.history['acc']) == 10)
assert(history.history['val_acc'][-1] > 0.92)
assert(history.history['val_acc'][-1] > 0.85)
print('Tests passed.')

##############################################
### Testing
##############################################
with open('test.p', 'rb') as f:
    data_test = pickle.load(f)

x_test = data_test['features']
y_test = data_test['labels']

x_normalized_test = normalize_grayscale(x_test)
y_one_hot_test = label_binarizer.fit_transform(y_test)

metrics = model.evaluate(x_normalized_test, y_one_hot_test)
for i in range(len(model.metrics_names)):
    metric_name = model.metrics_names[i]
    metric_value = metrics[i]
    print('{}: {}'.format(metric_name, metric_value))
