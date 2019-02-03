#!/usr/bin/env python

##################################################
### Load Data (MNIST data comes as 28x28x1 images
##################################################
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNST_data/", reshape=False)
x_train, y_train           = mnist.train.images, mnist.train.labels
x_validation, y_validation = mnist.validation.images, mnist.validation.labels
x_test, y_test             = mnist.test.images, mnist.test.labels

assert(len(x_train) == len(y_train))
assert(len(x_validation) == len(y_validation))
assert(len(x_test) == len(y_test))

print()
print("Image Shape: {}".format(x_train[0].shape))
print()
print("Training Set:   {} samples".format(len(x_train)))
print("Validation Set: {} samples".format(len(x_validation)))
print("Test Set:       {} samples".format(len(x_test)))

### Pading images with 0s from 28x28 to 32x32
import numpy as np

x_train      = np.pad(x_train, ((0,0), (2,2), (2,2), (0,0)), 'constant')
x_validation = np.pad(x_validation, ((0,0), (2,2), (2,2), (0,0)), 'constant')
x_test       = np.pad(x_test,  ((0,0), (2,2), (2,2), (0,0)), 'constant')

print("Updated Image Shape: {}".format(x_train[0].shape))

##################################################
### Visualizing
##################################################
import random
import numpy as np
import matplotlib.pyplot as plt

index = random.randint(0, len(x_train))
image = x_train[index].squeeze()

plt.figure(figsize=(1,1))
plt.imshow(image, cmap="gray")
plt.show()
print(y_train[index])

##################################################
### Preprocessing
##################################################
from sklearn.utils import shuffle

x_train, y_train = shuffle(x_train, y_train)


##################################################
### Training
##################################################
import tensorflow as tf
import sys
sys.path.append("../")
from nn.LeNet import LeNet_5

EPOCHS = 400
BATCH_SIZE = 128

# x is for input images
x = tf.placeholder(tf.float32, (None, 32, 32, 1))
# y is for output labels
y = tf.placeholder(tf.int32, (None))
one_hot_y = tf.one_hot(y, 10)

# Training Pipeline
rate = 0.001

logits = LeNet_5(x)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)
loss_operation = tf.reduce_mean(cross_entropy)
training_operation = tf.train.AdamOptimizer(learning_rate = rate).minimize(loss_operation)

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()

def evaluate(x_data, y_data):
    num_examples = len(x_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = x_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y})
        total_accuracy += (accuracy * len(batch_x))

    return total_accuracy / num_examples

import pickle

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(x_train)

    print("Training...")
    print()
    epo = []
    for i in range(EPOCHS):
        x_train, y_train = shuffle(x_train, y_train)
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = x_train[offset:end], y_train[offset:end]
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y})

        validation_accuracy = evaluate(x_validation, y_validation)
        epo.append([i,validation_accuracy])
        print("EPOCH {} ...".format(i+1))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print()

    saver.save(sess, './lenet')
    print("Model saved")

    f = open("lenet.txt", "wb")
    pickle.dump(epo, f)

##################################################
### Evaluation
##################################################
with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('.'))

    test_accuracy = evaluate(x_test, y_test)
    print("Test Accuracy = {:.3f}".format(test_accuracy))
    
