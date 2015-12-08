#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: lancezhange
# @Date:   2015-08-12 11:58:50
# @Last Modified by:   lancezhange
# @Last Modified time: 2015-12-08 21:21:48

# #######################################
# 模型训练主文件
# 可以插拔不同的模型和特征的定义
# ######################################


import sys
import cPickle
import logging
import logging.config
from utils import prepareData


# 导入配置
from smokeDetection_config import config
from importlib import import_module

feature = import_module(config.get("feature", "feature_file"))
model = import_module(config.get("model", "algo_file"))

logging.config.fileConfig("logger.conf")
logger = logging.getLogger("smoke_logger")


# 默认的训练数据
if config.getint("model", "isOverallModel"):
    training_path_a = config["data"]["image_train_positive"]
    training_path_b = config["data"]["image_train_negative"]
    model_file = config["model"]["overallModel_file"]
else:
    training_path_a = config["data"]["imagePart_train_positive"]
    training_path_b = config["data"]["imagePart_train_negative"]
    model_file = config["model"]["localModel_file"]


def main(training_path_a, training_path_b):
    '''
    Main function. Trains a classifier.

    Args:
      training_path_a (str): directory containing sample images
                             of class positive(1).
      training_path_b (str): directory containing sample images
                             of class positive(0).
    '''
    logger.info("prepareing data")
    x_train, x_test, y_train, y_test = prepareData(
        training_path_a,
        training_path_b,
        feature.getFeature)
    logger.info("prepare data done.")
    logger.info('Training classifier')
    classifier = model.getModel(x_train, x_test, y_train, y_test)
    logger.info("done. Saving model to file " + model_file)
    with open(model_file, 'wb') as fid:
        cPickle.dump(classifier, fid)
    logger.info("model saved successfully")

if __name__ == '__main__':
    if len(sys.argv) == 3:
        training_path_a = sys.argv[1]
        training_path_b = sys.argv[2]
    main(training_path_a, training_path_b)
