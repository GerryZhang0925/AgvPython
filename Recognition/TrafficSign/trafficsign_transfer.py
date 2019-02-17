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

def DownLoadData():
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
import time
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from scipy.misc import imread
import tensorflow as tf

def LoadData():
    with open('train.p', 'rb') as f:
        data = pickle.load(f)

        #x_train, x_val, y_train, y_val = train_test_split(data['features'], data['labels'], test_size=0.33, random_state=0)
    print('Tests passed.')
    return data

######################################################
### Global Constants
######################################################
NB_CLASSES = 43     # number of traffic sign classes
EPOCHS     = 100    # number of epochs
BATCH_SIZE = 128    # batch size

######################################################
### Retrain AlexNet
######################################################
import sys
sys.path.append("../")
from nn.AlexNet import AlexNet

def TrainAlexNet(data):

    print("Training AlexNet....")

    x_train, x_val, y_train, y_val = train_test_split(data['features'], data['labels'], test_size=0.33, random_state=0)
    
    features = tf.placeholder(tf.float32, (None, 32, 32, 3))
    labels = tf.placeholder(tf.int64, None)
    resized = tf.image.resize_images(features, (227, 227))

    # Returns the second final layer of the AlexNet model,
    # this allows us to redo the last layer for the traffic signs
    # model.
    fc7 = AlexNet(resized, feature_extract=True)
    fc7 = tf.stop_gradient(fc7)
    shape = (fc7.get_shape().as_list()[-1], NB_CLASSES)
    fc8W = tf.Variable(tf.truncated_normal(shape, stddev=1e-2))
    fc8b = tf.Variable(tf.zeros(NB_CLASSES))
    logits = tf.matmul(fc7, fc8W) + fc8b
    
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
    loss_op = tf.reduce_mean(cross_entropy)
    opt = tf.train.AdamOptimizer()
    train_op = opt.minimize(loss_op, var_list=[fc8W, fc8b])
    init_op = tf.global_variables_initializer()
    
    preds = tf.arg_max(logits, 1)
    accuracy_op = tf.reduce_mean(tf.cast(tf.equal(preds, labels), tf.float32))
    
    def eval_on_data(x, y, sess):
        total_acc  = 0
        total_loss = 0
        
        for offset in range(0, x.shape[0], BATCH_SIZE):
            end = offset + BATCH_SIZE
            x_batch = x[offset:end]
            y_batch = y[offset:end]
            
            loss, acc = sess.run([loss_op, accuracy_op], feed_dict={features: x_batch, labels: y_batch})
            total_loss += (loss * x_batch.shape[0])
            total_acc  += (acc * x_batch.shape[0])
        
        return total_loss/x.shape[0], total_acc/x.shape[0]

    log = []
    
    with tf.Session() as sess:
        sess.run(init_op)

        for i in range(EPOCHS):
            # training
            x_train, y_train = shuffle(x_train, y_train)
            t0 = time.time()
            total_acc  = 0
            total_loss = 0
            for offset in range(0, x_train.shape[0], BATCH_SIZE):
                end = offset + BATCH_SIZE
                sess.run(train_op, feed_dict={features: x_train[offset:end], labels: y_train[offset:end]})
                loss, acc = sess.run([loss_op, accuracy_op], feed_dict={features: x_train[offset:end], labels: y_train[offset:end]})
                total_loss += (loss * (end-offset))
                total_acc  += (acc * (end-offset))

            total_loss = total_loss/x_train.shape[0]
            total_acc  = total_acc/x_train.shape[0]
                
            val_loss, val_acc = eval_on_data(x_val, y_val, sess)
            print("Epoch", i+1)
            print("Time: %.3f seconds" % (time.time() - t0))
            print("Training Loss =", total_loss)
            print("Training Accuracy =", total_acc)
            print("Validation Loss =", val_loss)
            print("Validation Accuracy =", val_acc)
            print("")
            log.append([i+1, time.time() - t0, total_loss, total_acc, val_loss, val_acc])

    return log

######################################################
### Retrain VGG16
######################################################
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.layers import Input, Flatten, Dense, Dropout
from sklearn.preprocessing import LabelBinarizer
import matplotlib.pyplot as plt

def TrainVgg16(data):

    data['features'], data['labels'] = shuffle(data['features'], data['labels'])
    
    print("Training VggNet....")
    #model_vgg16 = VGG16(weights=None, include_top=False, input_shape=(32, 32, 3))
    model_vgg16 = VGG16(weights = "imagenet", include_top=False, input_shape=(32, 32, 3))
    print(model_vgg16.summary())
    #for layer in model_vgg16.layers:
    #    layer.trainable = False 

    label_binarizer = LabelBinarizer()
    y = data['labels']
    y_one_hot = label_binarizer.fit_transform(y)

    print(data['features'].shape)
    print(y_one_hot.shape)

    #index = 1000
    #image = data['features'][index]
    #plt.figure(figsize=(1,1))
    #plt.imshow(image)
    #print(y_one_hot[index])
    #plt.show()

    # Create the final model
    model = Sequential()
    model.add(model_vgg16)
    model.add(Flatten())
    model.add(Dense(512, activation='relu', name='fc1'))
    model.add(Dropout(0.5))
    model.add(Dense(512, activation='relu', name='fc2'))
    model.add(Dense(NB_CLASSES, activation='softmax', name='predictions'))
    print(model.summary())

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    history = model.fit(data['features'], y_one_hot, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_split=0.33, verbose=2)

    return history

######################################################
### Retrain ResNet50
######################################################
from keras.applications.resnet50 import ResNet50
from keras.models import Sequential

def TrainResNet50(data):

    data['features'], data['labels'] = shuffle(data['features'], data['labels'])
    
    print("Traing ResNet50....")
    #model_resnet50 = ResNet50(weights=None, include_top=False, input_shape=(32, 32, 3))
    model_resnet50 = ResNet50(weights='imagenet', include_top=False, input_shape=(32, 32, 3))
    print(model_resnet50.summary())
    #for layer in model_resnet50.layers:
    #    layer.trainable = False 

    label_binarizer = LabelBinarizer()
    y = data['labels']
    y_one_hot = label_binarizer.fit_transform(y)

    # Create the final model
    model = Sequential()
    model.add(model_resnet50)
    model.add(Flatten())
    model.add(Dense(2048, activation='relu', name='fc1'))
    model.add(Dropout(0.5))
    model.add(Dense(2048, activation='relu', name='fc2'))
    model.add(Dense(NB_CLASSES, activation='softmax', name='predictions'))
    print(model.summary())

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    history = model.fit(data['features'], y_one_hot, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_split=0.33, verbose=2)

    return history

######################################################
### Retrain Inception3
######################################################
from keras.applications.inception_v3 import InceptionV3
from keras.layers import GlobalMaxPooling2D, GlobalAveragePooling2D
import scipy
from scipy import misc

def TrainInceptionV3(data):

    data['features'], data['labels'] = shuffle(data['features'], data['labels'])
    
    print("Traing InceptionV3....")
    #model_inception_v3 = InceptionV3(weights=None, include_top=False, input_shape=(128, 128, 3))
    model_inception_v3 = InceptionV3(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
    print(model_inception_v3.summary())
    #for layer in model_inception_v3.layers:
    #    layer.trainable = False 

    big_x_train = np.array([scipy.misc.imresize(data['features'][i], (128, 128, 3)) 
                            for i in range(0, len(data['features']))]).astype('float32')
    
    label_binarizer = LabelBinarizer()
    y = data['labels']
    y_one_hot = label_binarizer.fit_transform(y)

    # Create the final model
    model = Sequential()
    model.add(model_inception_v3)
    model.add(GlobalAveragePooling2D())
    model.add(Dropout(0.5))
    model.add(Dense(NB_CLASSES, activation='softmax'))
    print(model.summary())

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    history = model.fit(big_x_train, y_one_hot, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_split=0.33, verbose=2)

    return history

import pickle

if __name__ == "__main__":
    DownLoadData()
    data = LoadData()

#    log = TrainAlexNet(data)
#    with open("alexnet.pkl", "wb") as f:
#        pickle.dump(log, f)

#    hist = TrainVgg16(data)
#    with open("vgg16-lock.pkl", "wb") as f:
#        pickle.dump(hist, f)
        
    hist = TrainResNet50(data)
    with open("resnet50.pkl", "wb") as f:
        pickle.dump(hist, f)

    hist = TrainInceptionV3(data)
    with open("incepv3.pkl", "wb") as f:
        pickle.dump(hist, f)
        
