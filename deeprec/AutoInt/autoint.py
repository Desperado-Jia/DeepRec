"""
Created on 2019/04/05 23:55
author: Tong Jia
email: cecilio.jia@gmail.com
description:
    An implement of AutoInt: Automatic Feature Interaction Learning via Self-Attentive Neural Networks (AutoInt).
    See algorithm and hyperparameter details:
        [Song et al., 2018](https://arxiv.org/pdf/1810.11921.pdf)
    The algorithm is developed with TensorFlow Estimator based on TensorFlow 1.12.0 version.
"""
import tensorflow as tf

def input_fn(filenames,
             delimiter,
             separator,
             batch_size,
             epochs,
             shuffle=True,
             buffer_size=100000,
             num_parallel_calls=4,
             dtype=tf.float32,
             name_feat_inds="inds",
             name_feat_vals="vals"):
    """
    The input function for loading multi-field sparse dataset. The columns are
    separeted by argument delimiter(e.g. " ") with the following schema (libsvm format):
    <label> <index 1>:<value 1> ... <index j>:<value j> ... <index d>:<value d> (d is the field size of dataset)
    e.g.
        0 0:0.732349 2:1 41:1 59:1 66:1 77:1 85:1 88:1 89:1 92:1 93:1 98:1 106:1
    where:
        delimiter is always " " in txt file format, "," in csv file format;
        separator is always equal to ":".
    Note:
        1. The input function is only used for dataset with one-hot active value in each field.
        2. In each line, the order of all fields must be fixed.
        3. The input function can be used for binary classification and regression task:
            binary classification: <label> in {0, 1};
            multi classification: <label> in {0, 1, ..., K}
            regression: <label> in (-inf, inf).

    Parameters
    ----------
    :param filenames: list
        A list of string, containing one or more paths of filenames.
    :param delimiter: str
        A str, separating consecutive <index j>:<value j> pairs in data files.
    :param separator: str
        A str, seprating feature index(left part of key-value pair) and feature value(right part of key-value pair) in one specific pair.
    :param batch_size: int
        An integer scalar, representing the number of consecutive elements of this dataset to combine in a single batch.
    :param epochs: int
        An integer scalar, representing the number of times the dataset should be repeated.
    :param shuffle: bool, optional
        A boolean(defaults to True), instructing whether to randomly shuffle the elements of this dataset.
    :param buffer_size: int, optional
        An integer scalar(defaults to 100000), denoting the number of bytes to buffer.
    :param num_parallel_calls: int, optional
        An integer scalar(defaults to 4), representing the number elements to process in parallel.
    :param dtype: tf.Dtype, optional
        A tf.DType(defaults to tf.float32), representing the numeric type of values. it always takes value from [tf.float32, tf.float64].
    :param name_feat_inds: str, optional
        A string, representing the name of feature indices in return dict.
    :param name_feat_vals: str, optional
        A string, representing the name of feature values in return dict.

    Returns
    -------
    :return: dict
        A dict of two Tensors, representing features(including feature indices and feature values) in a single batch.
        {
            <name_feat_inds>: tf.Tensor of feature indices in shape of (None, field_size),
            <name_feat_vals>: tf.Tensor of feature values in shape of (None, field_size)
        }
    :return: Tensor
        A Tensor in shape of (None), representing labels in a single batch.
    """
    def map_func(line):
        columns = tf.string_split(source=[line], delimiter=delimiter, skip_empty=False).values
        label = tf.string_to_number(string_tensor=columns[0], out_type=dtype)
        splits = tf.string_split(source=columns[1: ], delimiter=separator, skip_empty=False)
        feats = tf.reshape(
            tensor=splits.values, shape=splits.dense_shape
        ) # A tensor in shape of (field_size, 2), the first column contains feature indices, the second column contains feature values
        inds, vals = tf.split(value=feats, num_or_size_splits=2, axis=1) # Two tensors in shape of (field_size, 1)
        inds = tf.reshape(tensor=tf.string_to_number(string_tensor=inds, out_type=tf.int32), shape=[-1]) # A tensor in shape of (field_size)
        vals = tf.reshape(tensor=tf.string_to_number(string_tensor=vals, out_type=dtype), shape=[-1]) # A tensor in shape of (field_size)
        return {name_feat_inds: inds, name_feat_vals: vals}, label

    dataset = tf.data.TextLineDataset(filenames=filenames, buffer_size=buffer_size). \
        map(map_func=map_func, num_parallel_calls=num_parallel_calls)
    if (shuffle == True):
        dataset = dataset.shuffle(buffer_size=buffer_size)
    dataset = dataset. \
        repeat(count=epochs). \
        batch(batch_size=batch_size, drop_remainder=False)
    dataset = dataset.prefetch(buffer_size=1)
    iterator = dataset.make_one_shot_iterator()
    batch_feats, batch_labels = iterator.get_next()
    return batch_feats, batch_labels


def model_fn(features, labels, mode, params):
    # ----------Declare all hyperparameters from params----------
    task = params["task"]
    field_size = params["field_size"]
    feat_size = params["feat_size"]
    embed_size = params["embed_size"]

    dtype = params["dtype"]
    name_feat_inds = params["name_feat_inds"]
    name_feat_vals = params["name_feat_vals"]
    reuse = params["reuse"]
    seed = params["seed"]

