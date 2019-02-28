# coding: utf-8
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import math
from PIL import Image, ImageEnhance
import matplotlib
import paddle
import sys
import paddle.fluid as fluid
basedir = os.path.abspath(os.path.dirname(__file__))
# 模型地址
params_dirname = basedir + "\\model"
try:
    from paddle.fluid.contrib.trainer import *
    from paddle.fluid.contrib.inferencer import *
except ImportError:
    print(
        "In the fluid 1.0, the trainer and inferencer are moving to paddle.fluid.contrib",
        file=sys.stderr)
    from paddle.fluid.trainer import *
    from paddle.fluid.inferencer import *

DATA_DIM=64*64*1
def infer_func():
    x = fluid.layers.data(name='x', shape=[DATA_DIM], dtype='float32')
    h1=fluid.layers.fc(input=x, size=4, act='relu')
    y_predict = fluid.layers.fc(input=h1, size=2, act='softmax')
    return y_predict

def inference(image_path):
    # prepare data
    def load_image(img_path):
        img = Image.open(img_path)
        img = img.resize((64, 64), Image.LANCZOS)
        img = np.array(img)
        data=img.reshape(-1)
        data=data/255.0
        return data
    data = load_image(image_path)#归一化图像
    temp_data=np.asarray(data)
    temp_data=temp_data.reshape((len(temp_data),1))
    temp_data1=np.asarray(temp_data.T).astype("float32")
    exe=fluid.Executor(fluid.CPUPlace())
    exe.run(fluid.default_startup_program())
    print(params_dirname)
    aaa = Inferencer(infer_func=infer_func, param_path="G:\OtherProgTool\pycharm_pro\cat_dog\model", place=fluid.CPUPlace())
    result = aaa.infer({'x': temp_data1})

    return result
# result=inference('1.jpg')
# print("1识别结果为：",result)
# result=inference('2.jpg')
# print("2识别结果为：",result)




