#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: lancezhange
# @Date:   2015-08-21 15:16:43
# @Last Modified by:   lancezhange
# @Last Modified time: 2015-08-21 15:41:35

from sklearn import svm
from sklearn import decomposition
from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
import numpy as np


def getModel(x_train, x_test, y_train, y_test, print_metrics=True):
    '''
    Trains a classifier. We will use PCA to reduce dimensionality,
    and then train a SVM model. The dimensionality of the PCA will
    be set using GridSearchSV.

    Args:
      x_train (array of array): feature arrays of training data.
      x_test (array of array): feature arrays of testing data.
      y_train (array ): label arrays of training data.
      y_test (array ): label arrays of testing data.

    Returns:
      A classifier (sklearn.svm.SVC).
    '''
    pca = decomposition.PCA()
    svm_classifier = svm.SVC()
    pipe = Pipeline(steps=[("pca", pca), ("svm", svm_classifier)])
    pca.fit(x_train)
    n_components = [10, 20, 40, 60]
    gamma = [0.01, 0.001, 0.0001]
    Cs = [1, 10, 100, 1000]
    classifier = GridSearchCV(pipe,
                              dict(pca__n_components=n_components,
                                   svm__C=Cs,
                                   svm__gamma=gamma))
    classifier.fit(x_train, y_train)
    if print_metrics:
        print("parameters")
        print classifier.best_params_
        print("PCA with SVM score :\n%s\n" % (
            metrics.classification_report(
                y_test,
                classifier.predict(x_test))))
    return classifier
