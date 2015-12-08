#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: lancezhange
# @Date:   2015-08-12 14:49:21
# @Last Modified by:   lancezhange
# @Last Modified time: 2015-08-21 17:24:43

# 整体模型测试


import cPickle
import sys
import os
import logging
import logging.config
from utils import process_image_file
from smokeDetection_config import config

from importlib import import_module
feature = import_module(config.get("feature", "feature_file"))

# 阈值
threshold = config.getfloat("model", "threshold")

logging.config.fileConfig("logger.conf")
logger = logging.getLogger("smoke_logger")

if __name__ == '__main__':
    if(len(sys.argv) != 2):
        image_path = config.get("data", "image_eval_dir")
    else:
        image_path = sys.argv[1]

    # 全局模型
    overallModel_file = config["model"]["overallModel_file"]
    with open(overallModel_file, 'rb') as fid:
        classifier = cPickle.load(fid)
        logger.info("overall model imported successfully.")

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
    else:
        logger.error("not valied image/dir path!")
