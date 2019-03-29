#!/usr/bin/env python

print(__doc__)

import numpy as np
import pylab as pl
from sklearn import svm

# we create 40 separable points
np.random.seed(0)
x = np.r_[np.random.randn(20, 2) - [2, 2], np.random.randn(20, 2) + [2, 2]]
y = [0] * 20 + [1] * 20

# fit the model
clf = svm.SVC(kernel='linear')
clf.fit(x, y)

# get the separating hyperplane
w = clf.coef_[0]
a = -w[0] / w[1]
xx = np.linspace(-5, 5)
yy = a * xx - (clf.intercept_[0]) / w[1]

# plot the parallels to the separating hyperplane that pass through the
# support vectors
b = clf.support_vectors_[0]
yy_down = a * xx + (b[1] - a*b[0])
b = clf.support_vectors_[-1]
yy_up   = a * xx + (b[1] - a*b[0])

print("w: ".format(w))
print("a: ".format(a))

#print("xx: ".format(xx))
#print("yy; ".format(yy))
print("support_vectors_: ".format(clf.support_vectors_))
print("clf.coef_: ".format(clf.coef_))

# switching to the generic n-dimensional parameterrization of the
# hyperplane to the 20-specific equation of a line
pl.plot(xx, yy, 'k-')
pl.plot(xx, yy_down, 'k--')
pl.plot(xx, yy_up, 'k--')

pl.scatter(clf.support_vectors_[:,0], clf.support_vectors_[:,1],
           s=80, facecolors='none')
pl.scatter(x[:,0], x[:,1], c=y, cmap=pl.cm.Paired)
pl.axis('tight')
pl.show()
