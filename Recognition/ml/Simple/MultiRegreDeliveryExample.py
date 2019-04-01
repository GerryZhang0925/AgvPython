#!/usr/bin/env python
from numpy import genfromtxt
import numpy as np
from sklearn import datasets, linear_model

dataPath = r'Delivery.csv'
deliveryData = genfromtxt(dataPath, delimiter=',')

print("data")
print(deliveryData)

x = deliveryData[:, :-1]
y = deliveryData[:, -1]

print("x:{}".format(x))
print("y:{}".format(y))

regr = linear_model.LinearRegression()
regr.fit(x, y)

print("coefficients {}".format(regr.coef_))
print("intercept: {}".format(regr.intercept_))

xPred = [[102, 6]]
yPred = regr.predict(xPred)
print("predicted y: {}".format(yPred))
