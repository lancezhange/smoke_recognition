#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: lancezhange
# @Date:   2015-08-21 15:04:56
# @Last Modified by:   lancezhange
# @Last Modified time: 2015-08-21 15:06:37


from sklearn.ensemble import GradientBoostingClassifier
from sklearn import metrics


def getModel(x_train, x_test, y_train, y_test, print_metrics=True):
    '''
    Trains a classifier. Here we use Gradient Boosting.

    Args:
      x_train (array of array): feature arrays of training data.
      x_test (array of array): feature arrays of testing data.
      y_train (array ): label arrays of training data.
      y_test (array ): label arrays of testing data.

    Returns:
      A classifier (sklearn.ensemble.GradientBoostingClassifier).
    '''
    clf = GradientBoostingClassifier(
        n_estimators=100, learning_rate=1.0, random_state=0)
    classifier = clf.fit(x_train, y_train)
    if print_metrics:
        print('classifier score')
        print(metrics.classification_report(y_test,
                                            classifier.predict(x_test)))
    return classifier
