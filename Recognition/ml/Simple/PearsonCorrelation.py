#!/usr/bin/env python
import numpy as np
from astropy.units import Ybarn
import math

def computeCorrelation(x, y):
    xBar = np.mean(x)
    yBar = np.mean(y)
    SSR  = 0
    varX = 0
    varY = 0
    for i in range(0, len(x)):
        diffXXBar = x[i] - xBar
        diffYYBar = y[i] - yBar
        SSR       += (diffXXBar * diffYYBar)
        varX      += diffXXBar**2
        varY      += diffYYBar**2

    SST = math.sqrt(varX * varY)
    return SSR/SST

# Polynomial Regression
def polyfit(x, y, degree):
    results = {}

    coeffs = np.polyfit(x, y, degree)

    # Polynomial Coefficients
    results['polynomial'] = coeffs.tolist()

    # r-squared
    p = np.poly1d(coeffs)
    # fit values, and mean
    yhat  = p(x)                       # or [p(z) for
    ybar  = np.sum(y)/len(y)           # or sum(y)/len(y)
    ssreg = np.sum((yhat-ybar)**2)     # or sum([(yihat - ybar)**2 for yihat in yhat])
    print("ssreg:{}".format(ssreg))
    sstot = np.sum((y - ybar)**2)      # or sum([(yi - ybar)**2 for yi in y])
    print("sstot:{}".format(sstot))
    results['determination'] = ssreg / sstot

    print("results:{}".format(results))
    return results

testX = [1, 3, 8, 7, 9]
testY = [10, 12, 24, 21, 34]

print("r:{}".format(computeCorrelation(testX, testY)))
print("r^2:{}".format(computeCorrelation(testX, testY)**2))
print(polyfit(testX, testY, 1)["determination"])
