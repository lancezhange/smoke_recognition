#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: root
# @Date:   2015-08-25 15:44:32
# @Last Modified by:   lancezhange
# @Last Modified time: 2015-08-26 14:59:21

import cv2
import numpy as np
import logging
import logging.config
from array import array


logging.config.fileConfig("logger.conf")
logger = logging.getLogger("smoke_logger")


orb = cv2.ORB_create(1000)

image_path = 'image_train/smoke/building_smoke_0015.jpg'


def getFeature(image_path):
    global feature
    img0 = cv2.imread(image_path)
    if img0 is None:
        logger.info("not valid image path %s" % image_path)
        return np.empty(0)
    else:
        img = cv2.cvtColor(img0, cv2.COLOR_RGB2GRAY)
        kp = orb.detect(img, None)
        kp, des = orb.compute(img, kp)
        if des is None:
            return np.empty(0)
        else:
            feature.append(i for i in des.tolist())

getFeature(image_path)




# 计算每个关键点的描述
kp, des = orb.compute(img, kp)

if(os.path.isfile(image_path)):
        feature = process_image_file(image_path, feature.getFeature)
        proba = classifier.predict_proba(feature)
        logger.info("feature: %s", feature)
        logger.info(os.path.basename(image_path) +
                    " is 1 with proba %f" % proba[0][1])
    elif os.path.isdir(image_path):
        file_count = 0
        positive_count = 0
        for root, _, files in os.walk(image_path):
            for file_name in files:
                file_count = file_count + 1
                file_path = os.path.join(root, file_name)
                proba = classifier.predict_proba(
                    process_image_file(file_path, feature.getFeature))
                logger.info(os.path.basename(file_path) +
                            " is 1 with proba %f" % proba[0][1])
                if(proba[0][1] > threshold):
                    positive_count = positive_count + 1
        logger.info("%d be positive in total %d images" % (
            positive_count, file_count))
