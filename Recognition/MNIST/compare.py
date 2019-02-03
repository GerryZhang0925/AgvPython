#!/usr/bin/env python

import pickle
import matplotlib.pyplot as plt

f = open("lenet.txt", "rb")
lenet = pickle.load(f)

epo0 = []
acc0 = []
for epo, acc in lenet:
    epo0.append(epo)
    acc0.append(acc)

f = open("perceptron.txt", "rb")
perceptron = pickle.load(f)

epo1 = []
acc1 = []
for epo, acc in perceptron:
    epo1.append(epo)
    acc1.append(acc)

p1 = plt.plot(epo0, acc0, label = "w/ LeNet")
p2 = plt.plot(epo1, acc1, label = "w/ Perceptron")
plt.legend()
plt.title("Accuracy vs. Epoch (MNIST)")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.show()
