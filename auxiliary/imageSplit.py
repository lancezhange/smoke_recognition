#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: lancezhange
# @Date:   2015-08-20 08:55:44
# @Last Modified by:   lancezhange
# @Last Modified time: 2015-08-21 10:55:54

# 图片切割
# 单张图片或者目录均可


import os
import cv2
import uuid
import logging
import logging.config
logging.config.fileConfig("../logger.conf")

logger = logging.getLogger("smoke_logger")


# 图片所在路径
image_path = "../data_test_more/nosmoke"
# 存储路径
save_path = "../../imageProcessExamples/tmpdata/"
# 图片名称抬头
basename = "image"

# 切割图片大小设定
winH = 64
winW = 64
stepSize = 32


def sliding_window(image, stepSize=32, windowSize=(64, 64)):
    for y in xrange(0, image.shape[0] - windowSize[1] + 1, stepSize):
        for x in xrange(0, image.shape[1] - windowSize[0]+1, stepSize):
            yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])


def splitImage(image_file):
    image = cv2.imread(image_file)
    count = 0
    for (x, y, window) in sliding_window(
            image, stepSize, (winW, winH)):
        count = count + 1
        cv2.imwrite(
            save_path + basename + "_" +
            str(uuid.uuid3(uuid.NAMESPACE_DNS, str(count))) + ".jpg", window)

if __name__ == '__main__':
    if(os.path.isfile(image_path)):
        splitImage(image_path)
        logger.info("done.")
    elif os.path.isdir(image_path):
        for root, _, files in os.walk(image_path):
            for file_name in files:
                file_path = os.path.join(root, file_name)
                splitImage(file_path)
        logger.info("done")
    else:
        logger.error("not valid image path.")
