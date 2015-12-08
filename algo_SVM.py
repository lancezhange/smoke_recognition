#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: lancezhange
# @Date:   2015-08-21 11:12:11
# @Last Modified by:   lancezhange
# @Last Modified time: 2015-08-21 11:14:12


from sklearn import grid_search
from sklearn import svm
from sklearn import metrics


def getModel(x_train, x_test, y_train, y_test, print_metrics=True):
    '''
    Trains a classifier. Here we use SVM.

    Args:
      x_train (array of array): feature arrays of training data.
      x_test (array of array): feature arrays of testing data.
      y_train (array ): label arrays of training data.
      y_test (array ): label arrays of testing data.

    Returns:
      A classifier (sklearn.svm.SVC).
    '''
    # define the parameter search space
    parameters = {'kernel': ['linear', 'rbf'], 'C': [1, 10, 100, 1000],
                  'gamma': [0.01, 0.001, 0.0001]}
    # search for the best classifier within the search space and return it
    clf = grid_search.GridSearchCV(svm.SVC(), parameters).fit(x_train, y_train)
    classifier = clf.best_estimator_
    if print_metrics:
        print('Parameters:', clf.best_params_)
        print('Best classifier score')
        print(metrics.classification_report(y_test,
              classifier.predict(x_test)))
    return classifier
