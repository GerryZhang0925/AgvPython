#!/usr/bin/env python

#########################################
### Load Data
#########################################
from sklearn.model_selection import train_test_split
from six.moves import urllib
import os.path
import shutil
import cv2
import zipfile
import numpy

DEFAULT_SOURCE_URL = 'http://benchmark.ini.rub.de/Dataset/'

def urlretrieve_with_retry(url, filename=None):
    return urllib.request.urlretrieve(url, filename)

def maybe_download(filename, work_directory, source_url):
    if not os.path.exists(work_directory):
        os.mkdir(work_directory)

    filepath = os.path.join(work_directory, filename)
    if not os.path.exists(filepath):
        temp_file_name, _ = urlretrieve_with_retry(source_url)
        shutil.copy(temp_file_name, filepath)
        size = os.path.getsize()

        print('Successfully downloaded', filename, size, 'bytes.')
        
    return filepath

def extract_images(filename):
    print('Extracting', filename)
    images = []
    labels = []
    with zipfile.ZipFile(filename, "r") as zip_ref:
        zip_files = [files for files in zip_ref.filelist if files.filename.endswith('.ppm')]
        for file in zip_files:
            zip_ref.extract(file, '.')
            # Read Images
            im = cv2.imread(file.filename)
            im = cv2.resize(im, (32, 32))
            # Analyze Labels
            pathname = file.filename
            index = pathname.find('Images/')
            index += 7
            index2 = pathname.find('/', index)
            label = numpy.uint32(pathname[index:index2])
            images.append(im)
            labels.append(label)
        print("{} files extracted.".format(len(images)))
        zip_ref.close()
        
    return images, labels
    
def read_data_set(train_dir,
                  reshape=True,
                  source_url=DEFAULT_SOURCE_URL):

    # empty string check
    if not source_url:
        source_url = DEFUALT_SOURCE_URL
        
    TRAIN_IMAGES = 'GTSRB_Final_Training_Images.zip'
    TEST_IMAGES = 'GTSRB_Final_Test_Images.zip'

    # Open Training Data
    local_file = maybe_download(TRAIN_IMAGES, train_dir, source_url + TRAIN_IMAGES)
    x_orig, y_orig = extract_images(local_file)

    assert(len(x_orig) == len(y_orig))
    (x_train, x_test, y_train, y_test) = train_test_split(
        x_orig, y_orig, test_size = 0.2, random_state=0)

    return x_train, x_test, y_train, y_test

x_train, x_test, y_train, y_test = read_data_set("GTSRB/", reshape=False)

print()
print("Image Shape: {}".format(x_train[0]))
print()
print("Training Set:   {} samples".format(len(x_train)))
print("Test Set:       {} samples".format(len(x_test)))

##################################################
### Visualizing
##################################################
import random
import numpy as np
import matplotlib.pyplot as plt

index = random.randint(0, len(x_train))
image = x_train[index].squeeze()

plt.figure(figsize=(1,1))
plt.imshow(image)
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

EPOCHS = 10
BATCH_SIZE = 128

# x is for input images
x = tf.placeholder(tf.float32, (None, 32, 32, 3))
# y is for output labels
y = tf.placeholder(tf.int32, (None))
one_hot_y = tf.one_hot(y, 43)

# Training Pipeline
rate = 0.001

logits = LeNet_5(x, color_channel=3, output_classes=43)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = rate)
training_operation = optimizer.minimize(loss_operation)

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

        validation_accuracy = evaluate(x_test, y_test)
        epo.append([i,validation_accuracy])
        print("EPOCH {} ...".format(i+1))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print()

    saver.save(sess, './lenet')
    print("Model saved")

    f = open("lenet.txt", "wb")
    pickle.dump(epo, f)
