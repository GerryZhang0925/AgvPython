#!/usr/bin/env python
import numpy as np
from sklearn.datasets import load_digits
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import LabelBinarizer
from NeuralNetwork import NeuralNetwork
from sklearn.cross_validation import train_test_split

digits = load_digits()
x = digits.data
y = digits.target
x -= x.min() # normalize the values to bring them into the range 0-1
x /= x.max()

nn = NeuralNetwork([64,100,10], 'logistic')
x_train, x_test, y_train, y_test = train_test_split(x, y)
labels_train = LabelBinarizer().fit_transform(y_train)
labels_test  = LabelBinarizer().fit_transform(y_test)
print("start fitting")
nn.fit(x_train, labels_train, epochs=3000)
predictions = []
for i in range(x_test.shape[0]):
    o = nn.predict(x_test[i])
    predictions.append(np.argmax(o))
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))
