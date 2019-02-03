#!/usr/bin/env python

##################################################
### Load Data (MNIST data comes as 28x28x1 images
##################################################
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

mnist = input_data.read_data_sets("MNST_data/", reshape=True, one_hot=True)
x_train, y_train           = mnist.train.images, mnist.train.labels.astype(np.float32)
x_validation, y_validation = mnist.validation.images, mnist.validation.labels.astype(np.float32)
x_test, y_test             = mnist.test.images, mnist.test.labels.astype(np.float32)
assert(len(x_train) == len(y_train))
assert(len(x_validation) == len(y_validation))
assert(len(x_test) == len(y_test))

print()
print("Image Shape: {}".format(x_train[0].shape))
print()
print("Training Set:   {} samples".format(len(x_train)))
print("Validation Set: {} samples".format(len(x_validation)))
print("Test Set:       {} samples".format(len(x_test)))

##################################################
### Visualizing
##################################################
import random
import numpy as np
import matplotlib.pyplot as plt
import cv2

index = random.randint(0, len(x_train))
image = x_train[index].squeeze()
image = cv2.resize(image, (28, 28))

print(image.shape)
plt.figure(figsize=(1,1))
plt.imshow(image, cmap="gray")
plt.show()
print(y_train[index])

##################################################
### Preprocessing
##################################################
from sklearn.utils import shuffle

#x_train, y_train = shuffle(x_train, y_train)

##################################################
### Training
##################################################
import tensorflow as tf

EPOCHS = 400
BATCH_SIZE = 128
rate = 0.001

n_input = 784  # MNIST data input (img shape: 28*28)
n_classes = 10 # MNIST total classes (0-9 digits)

# Features and Labels
features = tf.placeholder(tf.float32, [None, n_input])
labels = tf.placeholder(tf.float32, [None, n_classes])

# Weights & bias
weights = tf.Variable(tf.random_normal([n_input, n_classes]))
bias = tf.Variable(tf.random_normal([n_classes]))

# Logits - xW + b
logits = tf.add(tf.matmul(features, weights), bias)

# Define loss and optimizer
learning_rate = tf.placeholder(tf.float32)
loss_operation = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
training_operation = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss_operation)

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()

def evaluate(x_data, y_data):
    num_examples = len(x_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = x_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={features: batch_x, labels: batch_y})
        total_accuracy += (accuracy * len(batch_x))

    return total_accuracy / num_examples

def batches(batch_size, features, labels):
    assert len(features) == len(labels)

    output_batches = []

    sample_size = len(features)
    print(sample_size)
    for start_i in range(0, sample_size, batch_size):
        end_i = start_i + batch_size
        batch = [features[start_i:end_i], labels[start_i:end_i]]
        output_batches.append(batch)

    return output_batches

import pickle

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(x_train)

    print("Training...")
    print()
    epo = []
    train_batches = batches(BATCH_SIZE, x_train, y_train)
        
    for i in range(EPOCHS):

        # Loop over all batches
        for batch_features, batch_labels in train_batches:
            sess.run(training_operation,
                     feed_dict={features: batch_features,
                                labels: batch_labels,
                                learning_rate: rate})
            
        validation_accuracy = evaluate(x_validation, y_validation)
        epo.append([i,validation_accuracy])
        print("EPOCH {} ...".format(i+1))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print()

    saver.save(sess, './perceptron')
    print("Model saved")

    f = open("perceptron.txt", "wb")
    pickle.dump(epo, f)
