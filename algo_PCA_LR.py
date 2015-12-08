#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: lancezhange
# @Date:   2015-08-21 15:16:43
# @Last Modified by:   lancezhange
# @Last Modified time: 2015-08-21 15:31:33

from sklearn import linear_model
from sklearn import decomposition
from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
import numpy as np


def getModel(x_train, x_test, y_train, y_test, print_metrics=True):
    '''
    Trains a classifier. We will use PCA to reduce dimensionality,
    and then train a LR model. The dimensionality of the PCA will
    be set using GridSearchSV.

    Args:
      x_train (array of array): feature arrays of training data.
      x_test (array of array): feature arrays of testing data.
      y_train (array ): label arrays of training data.
      y_test (array ): label arrays of testing data.

    Returns:
      A classifier (sklearn.linear_model.LogisticRegression).
    '''
    pca = decomposition.PCA()
    logistic_classifier = linear_model.LogisticRegression(C=100.0)
    pipe = Pipeline(steps=[("pca", pca), ("logistic", logistic_classifier)])
    pca.fit(x_train)
    n_components = [10, 20, 40, 60]
    Cs = np.logspace(-4, 4, 3)
    classifier = GridSearchCV(pipe,
                              dict(pca__n_components=n_components,
                                   logistic__C=Cs))
    classifier.fit(x_train, y_train)
    if print_metrics:
        print("parameters")
        print classifier.best_params_
        print("PCA with LR score :\n%s\n" % (
            metrics.classification_report(
                y_test,
                classifier.predict(x_test))))
    return classifier
