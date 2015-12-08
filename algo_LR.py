#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: lancezhange
# @Date:   2015-08-18 11:28:02
# @Last Modified by:   lancezhange
# @Last Modified time: 2015-08-22 11:11:07

from sklearn import linear_model, metrics


def getModel(x_train, x_test, y_train, y_test, print_metrics=True):
    '''
    Trains a classifier. Here we use LR.

    Args:
      x_train (array of array): feature arrays of training data.
      x_test (array of array): feature arrays of testing data.
      y_train (array ): label arrays of training data.
      y_test (array ): label arrays of testing data.

    Returns:
      A classifier (sklearn.linear_model.LogisticRegression).
    '''
    logistic_classifier = linear_model.LogisticRegression(C=100.0)
    logistic_classifier.fit(x_train, y_train)
    if print_metrics:
        print("Logistic regression :\n%s\n" % (
            metrics.classification_report(
                y_test,
                logistic_classifier.predict(x_test))))
    return logistic_classifier
