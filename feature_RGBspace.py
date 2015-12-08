#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: lancezhange
# @Date:   2015-08-12 14:47:33
# @Last Modified by:   lancezhange
# @Last Modified time: 2015-08-21 11:23:14


from __future__ import division
import numpy as np


def getFeature(image, blocks=5):
    '''Given an cv2 image object it returns its feature vector.

    Args:
      image (ndarray):  image array
      blocks (int, optional): number of block to subdivide the RGB space into.

    Returns:
      numpy array
    '''
    feature = np.zeros(blocks*blocks*blocks)
    width, height, channel = image.shape
    pixel_count = width*height
    r = ((image[:, :, 2].reshape(pixel_count, 1)) / (256/blocks)).astype("int")
    g = ((image[:, :, 1].reshape(pixel_count, 1)) / (256/blocks)).astype("int")
    b = ((image[:, :, 0].reshape(pixel_count, 1)) / (256/blocks)).astype("int")
    result = r + g*blocks + b*blocks*blocks
    unique, counts = np.unique(result, return_counts=True)
    feature[unique] = counts
    return feature/(pixel_count)
