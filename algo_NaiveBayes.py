#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: lancezhange
# @Date:   2015-08-21 14:55:12
# @Last Modified by:   lancezhange
# @Last Modified time: 2015-08-21 15:13:00


from sklearn.naive_bayes import GaussianNB
from sklearn import metrics


def getModel(x_train, x_test, y_train, y_test, print_metrics=True):
    '''
    Trains a classifier. Here we use Naive Bayes.

    Args:
      x_train (array of array): feature arrays of training data.
      x_test (array of array): feature arrays of testing data.
      y_train (array ): label arrays of training data.
      y_test (array ): label arrays of testing data.

    Returns:
      A classifier (sklearn.naive_bayes.GaussianNB).
    '''
    gnb = GaussianNB()
    classifier = gnb.fit(x_train, y_train)
    if print_metrics:
        print('classifier score')
        print(metrics.classification_report(y_test,
              classifier.predict(x_test)))
    return classifier
