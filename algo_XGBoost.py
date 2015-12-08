#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: lancezhange
# @Date:   2015-08-21 16:18:56
# @Last Modified by:   lancezhange
# @Last Modified time: 2015-08-21 22:18:10

from xgbWrapper.xgb2sklearn import XGBoostClassifier


def getModel(x_train, x_test, y_train, y_test, print_metrics=True):
    '''
    Trains a classifier. We use xgboost here.

    Args:
      x_train (array of array): feature arrays of training data.
      x_test (array of array): feature arrays of testing data.
      y_train (array ): label arrays of training data.
      y_test (array ): label arrays of testing data.

    Returns:
      A classifier (sklearn.linear_model.LogisticRegression).
    '''

    # dtrain = xgb.DMatrix(x_train, label=y_train)
    # dtest = xgb.DMatrix(x_test, label=y_test)
    # param = {'bst:max_depth': 2, 'bst:eta': 1,
    #          'silent': 1, 'objective': 'binary:logistic'}
    # param['nthread'] = 4
    # plst = param.items()
    # plst += [('eval_metric', 'auc')]
    # num_round = 10
    # whatchlist = [(dtest, 'eval'), (dtrain, 'train')]
    # bst = xgb.train(plst, dtrain, num_round, whatchlist)
    # return bst
    xgb = XGBoostClassifier()
    xgb.set_params(
        object="binary:logistic",
        silent=0, nthread=4)
    xgb.fit(x_train, y_train)
    return xgb
