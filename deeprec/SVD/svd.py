"""
Created on 2019/04/09 22:04
author: Tong Jia
email: cecilio.jia@gmail.com
description:
    An implement of Matrix Factorization Techniques for Recommender Systems (SVD).
    See algorithm and hyperparameter details:
        [Koren et al., 2007](https://www.ime.usp.br/~jstern/miscellanea/seminars/nnmatrix/Koren07.pdf)
    The algorithm is developed with TensorFlow Estimator based on TensorFlow 1.12.0 version.
"""
import tensorflow as tf

