#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Author: lancezhange
# @Date:   2015-08-18 13:48:14
# @Last Modified by:   lancezhange
# @Last Modified time: 2015-08-24 12:09:19


# 服务接口

import cPickle
import sys
import logging
import logging.config
import cv2
from utils import process_image, sliding_window
# 导入配置
from smokeDetection_config import config
from importlib import import_module


feature = import_module(config.get("feature", "feature_file"))

# 窗宽
winH = config.getint("window", "winH")
winW = config.getint("window", "winW")
stepSize = config.getint("window", "stepSize")

logging.config.fileConfig("logger.conf")
logger = logging.getLogger("smoke_logger")


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
            # logger.info("overall model imported successfully.")
        # 局部模型
        localModel_file = config["model"]["localModel_file"]
        with open(localModel_file, 'rb') as fid:
            classifier2 = cPickle.load(fid)
            # logger.info("overall model imported successfully.")

    image = cv2.imread(image_path)
    pred = classifier1.predict_proba(process_image(
        image, feature.getFeature))[0]
    # threshold_window = 1 - min(0.5, pred[1]) * 5 / 7.0  # 窗口上的阈值取决于整体上的概率
    threshold_window = 0.6
    result = list()
    for (x, y, window) in sliding_window(image, stepSize, (winW, winH)):
        prediction = classifier2.predict_proba(
            process_image(window, feature.getFeature))[0]
        if(prediction[1] >= threshold_window):
            result.append([x, y, x+winW, y+winH])
    print result
