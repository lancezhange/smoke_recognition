#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: lancezhange
# @Date:   2015-08-18 11:28:02
# @Last Modified by:   lancezhange
# @Last Modified time: 2015-08-21 14:01:52

from __future__ import division
import cv2
import numpy as np
from skimage.feature import hog


def getFeature(image, blocks=5):
    '''
    Given an cv2 Image object it returns its feature vector.

    Args:
      image (ndarray): image to process.
      blocks (int, optional): number of block to subdivide the RGB space into.

    Returns:
      numpy array.
    '''
    image_resized = cv2.resize(image, (64, 64))
    imgray = cv2.cvtColor(image_resized, cv2.COLOR_RGB2GRAY)
    feature = np.zeros(144)
    feature_hog = hog(
        imgray, orientations=9,
        pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualise=False)
    feature[:] = feature_hog
    return feature
