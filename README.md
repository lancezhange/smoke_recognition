### 图片中的烟雾检测


#### 依赖
参见　requirements.txt

#### 训练模型
```{python}
python smokeDetection.py image_train/smoke image_train/nosmoke
# 或者简写为
python smokeDetection.py
# 如上方式将使用默认的训练图片目录
```

#### 调用模型分类
```{python}
# 用全局模型进行分类
python smokeDetection_eval_whole.py [单张图片路径或者目录]
# 用局部模型进行分类
python smokeDetection_eval_local.py [单张图片路径或者目录]
# 综合局部模型和全局模型进行分类
python smokeDetection_eval_combine.py [单张图片路径]
```

#### 配置选项

`smokeDetection.ini` 文件中，可以进行一些全局配置，例如

- 特征和模型定义文件
- 是否是全局模型（在训练局部模型时，这一项请务必置0）
- 默认的训练图片正负样例所在目录
- 保存的模型文件名称
- 判为正例的阈值


`logger.conf` 文件中可以对日志进行配置


#### 扩展模型
用户可以自定义特征和模型（只需定义相应的 getFeature 和 getModel 方法，然后在配置文件中注明即可（请参考文件 `features/feature_RGBspace_HOG.py` 和 `algorithms/algo_LR.py`中的写法，并注意返回值的类型）。





