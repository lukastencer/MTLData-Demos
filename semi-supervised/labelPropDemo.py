"""
===============================================
Label Propagation over a classification dataset
===============================================

Label propagation over a dataset for a classification problem
generated using scikit-learn functions for dataset generation. Also
inculdes visualization of decision boundary.
"""
print(__doc__)

# Authors: Lukas Tencer <lukas.tencer@gmail.com>
# Licence: BSD

import time
from copy import copy
import numpy as np
import matplotlib.pyplot as plt
from sklearn.semi_supervised import label_propagation
from sklearn.datasets import make_circles, make_multilabel_classification, make_classification, make_blobs
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB


class Generators:
    CLF, CLF_MULTI, BLOBS, CICRLES = range(4)

# --- get data ---

# parameter setup
showOrig = True
n_samples = 200
n_classes = 3
classes = range(0, n_classes)
shuffle = False
n_labeled = 1
generator = Generators.BLOBS
clf = GaussianNB()
iterations = 20

# sample generation
if generator == Generators.CLF:
    X, y = make_classification(n_samples=n_samples, n_features=n_classes,
                               n_informative=n_classes - 0, n_redundant=0,
                               n_classes=n_classes, shuffle=shuffle)
elif generator == Generators.BLOBS:
    X, y = make_blobs(n_samples=n_samples, n_features=2, centers=n_classes)
elif generator == Generators.CICRLES:
    X, y = make_circles(n_samples=n_samples, shuffle=shuffle)

labels = -np.ones(n_samples)

for cls in classes:
    labels[np.where((cls == y))[0][0:n_labeled]] = cls
orig_labels = copy(labels)

plt.figure(figsize=(14.05, 4.3))
plt.ion()
plt.show()

for max_iter in range(1, iterations):

    ###########################################################################
    # Learn with LabelSpreading
    label_spread = label_propagation.LabelSpreading(kernel='knn', alpha=1.0,
                                                    max_iter=max_iter)
    label_spread.fit(X, labels)

    output_labels = copy(label_spread.transduction_)

    # fix for demo presentation
    temp_labels = copy(labels)
    temp_labels[labels == 0] = 1
    temp_labels[labels == 1] = 0
    label_spread.fit(X, temp_labels)
    output_labels_fix = copy(label_spread.transduction_)

    output_labels[output_labels == 0] = -1
    output_labels[output_labels_fix == 1] = 0

    ###########################################################################
    markers = ['ro', 'bo', 'co', 'mo', 'yo',
               'ko', 'r*', 'b*', 'c*', 'm*', 'y*', 'k*']

    h = .02  # step size in the mesh
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Plot output

    plt.subplot(1, 3, 1)
    for cls, marker in zip(classes, markers):

        plot, = plt.plot(X[y == cls, 0],
                         X[y == cls, 1], marker, alpha=0.3, ms=5)

        plot, = plt.plot(X[orig_labels == cls, 0],
                         X[orig_labels == cls, 1], marker, ms=8)
        plot, = plt.plot(X[orig_labels == cls, 0],
                         X[orig_labels == cls, 1], 'k+', ms=8)

    if showOrig:
        cX = X[np.logical_not(orig_labels == -1), 0:2]
        cy = orig_labels[np.logical_not(orig_labels == -1)]

        clf.fit(cX, cy)
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        cm = plt.cm.spring
        colorHandle = plt.contourf(xx, yy, Z, cmap=plt.cm.Paired,
                                   levels=np.linspace(Z.min(), Z.max(), 10),
                                   alpha=.9)

    plt.title("Raw data")

    plt.subplot(1, 3, 2)
    plot_unlabeled, = plt.plot(X[labels == -1, 0], X[labels == -1, 1], 'g.')
    output_label_array = np.asarray(output_labels)
    for cls, marker in zip(classes, markers):
        idx = np.where(output_label_array == cls)[0]
        plot, = plt.plot(X[idx, 0], X[idx, 1], marker)

    plt.title("Labels learned with Label Spreading (KNN)")

    plt.subplot(1, 3, 3)
    output_label_array = np.asarray(output_labels)
    for cls, marker in zip(classes, markers):
        idx = np.where(output_label_array == cls)[0]
        plot, = plt.plot(X[idx, 0], X[idx, 1], marker)

    cX = X[np.logical_not(output_labels == -1), 0:2]
    cy = output_labels[np.logical_not(output_labels == -1)]

    clf.fit(cX, cy)
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    # Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:,1]
    # use previous line for soft regions

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    cm = plt.cm.spring
    colorHandle = plt.contourf(xx, yy, Z, cmap=plt.cm.Paired,
                               levels=np.linspace(Z.min(), Z.max(), 10),
                               alpha=.9)

    plt.title("Decision Boundary given KNN Classifier")

    plt.subplots_adjust(left=0.07, bottom=0.07, right=0.93, top=0.90)
    # fig.canvas.draw()
    plt.draw()
    time.sleep(0.05)
    # plt.show()

raw_input("Press Enter to continue...")
