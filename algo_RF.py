#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: lancezhange
# @Date:   2015-08-21 11:15:05
# @Last Modified by:   lancezhange
# @Last Modified time: 2015-08-21 11:16:45


from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics


def getModel(x_train, x_test, y_train, y_test, print_metrics=True):
    '''
    Trains a classifier. Here we use RandomForest.

    Args:
      x_train (array of array): feature arrays of training data.
      x_test (array of array): feature arrays of testing data.
      y_train (array ): label arrays of training data.
      y_test (array ): label arrays of testing data.

    Returns:
      A classifier (sklearn.ensemble.RandomForestClassifier).
    '''
    clf = RandomForestClassifier(n_estimators=20)
    classifier = clf.fit(x_train, y_train)
    if print_metrics:
        print('classifier score')
        print(metrics.classification_report(
            y_test,
            classifier.predict(x_test)))
    return classifier
