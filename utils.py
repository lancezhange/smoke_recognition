#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: lancezhange
# @Date:   2015-08-17 23:50:10
# @Last Modified by:   lancezhange
# @Last Modified time: 2015-08-21 15:56:22

import os
import sys
import cv2
import numpy as np
import logging
import logging.config
from sklearn import cross_validation
from sklearn.preprocessing import normalize


logging.config.fileConfig("logger.conf")
logger = logging.getLogger("smoke_logger")


def process_directory(directory, getFeature):
    '''
    Returns an array of feature vectors for all the image files in a
    directory (and all its subdirectories). Symbolic links are ignored.

    Args:
      directory (str): directory to process.
      getFeature (function): feature-extract function.

    Returns:
      list of list of float: a list of feature vectors.
    '''
    training = []
    for root, _, files in os.walk(directory):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            img_feature = process_image_file(file_path, getFeature)
            if img_feature.size != 0:
                training.append(img_feature)
    return training


# 由于目录下可能有非图片文件，所以遇到不能读取的错误并不终止程序运行
def process_image_file(image_path, getFeature):
    '''
    Given an image path it returns its feature vector.

    Args:
      image_path (str): path of the image file to process.
      getFeature (function): feature-extract function.

    Returns:
      list of float: feature vector on success, None otherwise.
    '''
    image = cv2.imread(image_path)
    if image is not None:
        return process_image(image, getFeature)
    else:
        logger.warning("read None form %s! File broken or not exist."
                       % image_path, exc_info=True)
        return np.empty(0)


def process_image(image, getFeature):
    '''
    Given an image it returns its feature vector.

    Args:
      image (ndarray): path of the image file to process.
      getFeature (function): feature-extract function.


    Returns:
      list of float: feature vector on success, None otherwise.
    '''
    if type(image).__module__ == np.__name__ and image.size > 0:
        return getFeature(image)
    else:
        logger.warning("image is None or Empty!")
        return np.empty(0)


def prepareData(training_path_a, training_path_b, getFeature,
                test_size=0.2, normal=False):
    '''
    Prepare data for training and testing phase.
    training_path_a and training_path_b should be directory paths and
    each of them should not be a subdirectory of the other one.

    Args:
      training_path_a (str): directory containing sample images
                             of class positive(1).
      training_path_b (str): directory containing sample images
                             of class positive(0).
      getFeature (function): functtion for extracting features
                             from iamge data.

    Returns:
      train data and test data.
    '''
    if not os.path.isdir(training_path_a):
        logger.critical('%s is not a directory' % training_path_a)
        sys.exit()
    if not os.path.isdir(training_path_b):
        logger.critical('%s is not a directory' % training_path_b)
        sys.exit()
    training_a = process_directory(training_path_a, getFeature)
    training_b = process_directory(training_path_b, getFeature)

    data = training_a + training_b
    target = [1] * len(training_a) + [0] * len(training_b)

    x_train, x_test, y_train, y_test = cross_validation.train_test_split(
        data, target, test_size=test_size)
    if normal:
        x_train = normalize(x_train, axis=0, norm="l1")
        x_test = normalize(x_test, axis=0, norm="l1")
    return(x_train, x_test, y_train, y_test)


def sliding_window(image, stepSize=32, windowSize=(64, 64)):
    for y in xrange(0, image.shape[0] - windowSize[1] + 1, stepSize):
        for x in xrange(0, image.shape[1] - windowSize[0] + 1, stepSize):
            yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])
