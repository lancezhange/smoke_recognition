#!/bin/sh
# @Author: lancezhange
# @Date:   2015-08-20 08:28:38
# @Last Modified by:   lancezhange
# @Last Modified time: 2015-08-20 08:52:26

# 实现对目录下图片的批量序列命名

image_dir=""
cd $image_dir

a=1
for i in *.jpg; do
  new=$(printf "smoke%04d.jpg" "$a")  # 04 pad to length of 4
  mv -- "$i" "$new"
  let a=a+1
done
