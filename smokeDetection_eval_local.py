#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: lancezhange
# @Date:   2015-08-12 12:00:01
# @Last Modified by:   lancezhange
# @Last Modified time: 2015-08-21 11:31:02

# 局部模型测试

import cPickle
import sys
import os
import cv2
from utils import process_image, sliding_window
import logging
import logging.config

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


# 局部模型
localModel_file = config["model"]["localModel_file"]
with open(localModel_file, 'rb') as fid:
    classifier2 = cPickle.load(fid)
    logger.info("local model imported successfully.")


def checkImage(image_file):
    image = cv2.imread(image_file)
    smoke_count = 0
    window_count = 0
    for (x, y, window) in sliding_window(
            image, stepSize, (winW, winH)):
        window_count = window_count + 1
        prediction = classifier2.predict_proba(
            process_image(window, feature.getFeature))[0]
        if(prediction[1] >= threshold):
            smoke_count = smoke_count + 1
    logger.info("%d smoke in %d widnows of %s." % (
        smoke_count, window_count, os.path.basename(image_file)))


if __name__ == '__main__':
    if(len(sys.argv) != 2):
        # 默认的测试文件目录
        image_path = config.get("data", "image_eval_dir")
    else:
        image_path = sys.argv[1]
    if(os.path.isfile(image_path)):
        checkImage(image_path)
    elif os.path.isdir(image_path):
        for root, _, files in os.walk(image_path):
            for file_name in files:
                file_path = os.path.join(root, file_name)
                checkImage(file_path)
    else:
        logger.error("not valid image/dir path!")
