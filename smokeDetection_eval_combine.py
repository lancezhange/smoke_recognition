#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: lancezhange
# @Date:   2015-08-12 14:53:15
# @Last Modified by:   lancezhange
# @Last Modified time: 2015-08-24 13:00:24


# 综合整体模型和局部模型进行测试

import cPickle
import logging
import logging.config
import sys
import os
import cv2
import matplotlib.pyplot as plt
from utils import process_image, sliding_window


# 导入配置
from smokeDetection_config import config

from importlib import import_module
feature = import_module(config.get("feature", "feature_file"))

logging.config.fileConfig("logger.conf")
logger = logging.getLogger("smoke_logger")


# 阈值
threshold = config.getfloat("model", "threshold")


# 窗宽
winH = config.getint("window", "winH")
winW = config.getint("window", "winW")
stepSize = config.getint("window", "stepSize")


if __name__ == '__main__':
    if(len(sys.argv) != 2):
        logger.critical("请输入测试图片路径！")
        sys.exit()
    else:
        image_path = sys.argv[1]
        # 全局模型
        overallModel_file = config["model"]["overallModel_file"]
        with open(overallModel_file, 'rb') as fid:
            classifier1 = cPickle.load(fid)
            logger.info("overall model imported successfully.")
        # 局部模型
        localModel_file = config["model"]["localModel_file"]
        with open(localModel_file, 'rb') as fid:
            classifier2 = cPickle.load(fid)
            logger.info("overall model imported successfully.")

    image = cv2.imread(image_path)
    if image is None:
        logger.error("Read None from %s! file is broken or not exist"
                     % os.path.basename(image_path))
        sys.exit()

    smoke_count = 0
    window_count = 0
    pred = classifier1.predict_proba(process_image(
        image, feature.getFeature))[0]
    if(pred[1] > threshold):
        logger.info("predict %d with proba %f" % (1, pred[1]))
    else:
        logger.info("predict %d with proba %f" % (0, pred[0]))
    # threshold_window = 1 - min(0.5, pred[1]) * 5 / 7.0  # 窗口上的阈值取决于整体上的概率
    threshold_window = 0.98
    # threshold_window = 1 - max(0.4, min(pred[1], 0.6))
    logger.info("threshold of the window is: %f" % (threshold_window))
    image_copy = image.copy()
    for (x, y, window) in sliding_window(image, stepSize, (winW, winH)):
        window_count = window_count + 1
        prediction = classifier2.predict_proba(
            process_image(window, feature.getFeature))[0]
        if(prediction[1] >= threshold_window):
            smoke_count = smoke_count + 1
            cv2.rectangle(image_copy, (x, y), (x + int(winW * prediction[1]),
                                               y + int(winH * prediction[1])),
                          (0, 255, 0), 2)
            # cv2.rectangle(image_copy, (x, y), (x + winW,
            #                                    y + winH),
            #               (0, 255, 0), 2)
    logger.info("%d smoke found in total %d widnows." % (
        smoke_count, window_count))

    if plt.get_backend() == "Qt4Agg":
        plt.imshow(image_copy)
        plt.show()
