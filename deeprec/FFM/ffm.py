"""
Created on 2019/03/31 20:03
author: Tong Jia
email: cecilio.jia@gmail.com
description:
    An implementation of Field-aware Factorization Machines for CTR Prediction (FFM).
    See algorithm and hyperparameter details:
        [Juan et al., 2016](https://www.csie.ntu.edu.tw/~cjlin/papers/ffm.pdf)
    The algorithm is developed with TensorFlow Estimator based on TensorFlow 1.12.0 version.
"""
import tensorflow as tf
import tensorflow.contrib as tfc

def input_fn(filenames, delimiter, separator, batch_size, epochs):
    def map_func(line):
