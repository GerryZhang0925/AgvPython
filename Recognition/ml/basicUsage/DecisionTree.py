#!/usr/bin/env python

import random

# Make the dataset
def makeTerrainData(points=1000):
    # creat the toy dataset
    random.seed(42)
    grade = [random.random() for i in range(0, points)]
    bumpy = [random.random() for i in range(0, points)]
    error = [random.random() for i in range(0, points)]
    y = [round(grade[i] * bumpy[i] + 0.3 + 0.1*error[i]) for i in range(0, points)]
    for i in range(0, len(y)):
        if grade[i] > 0.8 or bumpy[i] > 0.8:
            y[i] = 1.0

    # split into train/test sets
    x = [[gg, ss] for gg, ss in zip(grade, bumpy)]
    split = int(0.75 * points)
    x_train = x[0:split]
    x_test  = x[split:]
    y_train = y[0:split]
    y_test  = y[split:]

    grade_sig = [x_train[i][0] for i in range(0, len(x_train)) if y_train[i] == 0]
    bumpy_sig = [x_train[i][1] for i in range(0, len(x_train)) if y_train[i] == 0]
    grade_pkg = [x_train[i][0] for i in range(0, len(x_train)) if y_train[i] == 1]
    bumpy_pkg = [x_train[i][1] for i in range(0, len(x_train)) if y_train[i] == 1]

    training_data = {"fast":{"grade": grade_sig, "bumpiness": bumpy_sig},
                     "slow":{"grade": grade_pkg, "bumpiness": bumpy_pkg}}

    grade_sig = [x_test[i][0] for i in range(0, len(x_test)) if y_test[i] == 0]
    bumpy_sig = [x_test[i][1] for i in range(0, len(x_test)) if y_test[i] == 0]
    grade_bkg = [x_test[i][0] for i in range(0, len(x_test)) if y_test[i] == 1]
    bumpy_bkg = [x_test[i][1] for i in range(0, len(x_test)) if y_test[i] == 1]

    test_data = {"fast":{"grade": grade_sig, "bumpiness": bumpy_sig},
                 "slow":{"grade": grade_bkg, "bumpiness": bumpy_pkg}}

    return x_train, y_train, x_test, y_test

import warnings
warnings.filterwarnings("ignore")

import matplotlib
#matplotlib.use('agg')

import matplotlib.pyplot as plt
import pylab as pl
import numpy as np

def prettyPicture(clf, x_test, y_test):
    x_min = 0.0
    x_max = 1.0
    y_min = 0.0
    y_max = 1.0

    # Plot the decision boundary, For that, we will assign a color to each point in the mesh
    h = .01  # step size in the mesh
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    z = z.reshape(xx.shape)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())

    plt.pcolormesh(xx, yy, z, cmap=pl.cm.seismic)

    grade_sig = [x_test[i][0] for i in range(0, len(x_test)) if y_test[i] == 0]
    bumpy_sig = [x_test[i][1] for i in range(0, len(x_test)) if y_test[i] == 0]
    grade_pkg = [x_test[i][0] for i in range(0, len(x_test)) if y_test[i] == 1]
    bumpy_bkg = [x_test[i][1] for i in range(0, len(x_test)) if y_test[i] == 1]

    plt.scatter(grade_sig, bumpy_sig, color='b', label='fast')
    plt.scatter(grade_pkg, bumpy_bkg, color='r', label='slow')
    plt.legend()
    plt.xlabel('bumpiness')
    plt.ylabel('grade')
    plt.show()

import base64
import json
import subprocess


#def output_image(name, format, bytes):
#    image_start = "BEGIN_IMAGE_f9825uweof8jw9fj4r8"
#    image_end   = "END_IMAGE_0238jfw08fjsiufhw8frs"
#    data = {}
#    data['name']   = name
#    data['format'] = format
#    data['bytes']  = base64.encodestring(bytes)
#    print image_start+json.dumps(data)+image_end

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
    
def classify(x_train, y_train):
    clf = DecisionTreeClassifier()
    clf.fit(x_train, y_train)

    return clf
    
def main():
    features_train, labels_train, features_test, labels_test = makeTerrainData()

    grade_fast = [features_train[i][0] for i in range(0, len(features_train)) if labels_train[i] == 0]
    bumpy_fast = [features_train[i][1] for i in range(0, len(features_train)) if labels_train[i] == 0]
    grade_slow = [features_train[i][0] for i in range(0, len(features_train)) if labels_train[i] == 1]
    bumpy_slow = [features_train[i][1] for i in range(0, len(features_train)) if labels_train[i] == 1]

    clf = classify(features_train, labels_train)

    pred = clf.predict(features_test)
    print(accuracy_score(pred, labels_test))
    
    prettyPicture(clf, features_test, labels_test)



if __name__ == "__main__":
    main()
