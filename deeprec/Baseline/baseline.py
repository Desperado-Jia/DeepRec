"""
Created on 2019/04/08 01:46
author: Tong Jia
email: cecilio.jia@gmail.com
description:
    An implement of Baseline model, nemely:
        * Logistic Regression for binary classification task;
        * Softmax Regression for multi classification task;
        * Linear Regression foe regression task.
    The algorithm is developed with TensorFlow Estimator based on TensorFlow 1.12.0 version.
"""
import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer, l2_regularizer

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
    output_size = params["output_size"]
    # field_size = params["field_size"]
    feat_size = params["feat_size"]
    use_global_bias = params["use_global_bias"]
    lamb = params["lamb"]
    optimizer = params["optimizer"]
    learning_rate = params["learning_rate"]
    dtype = params["dtype"]
    name_feat_inds = params["name_feat_inds"]
    name_feat_vals = params["name_feat_vals"]
    reuse = params["reuse"]
    seed = params["seed"]
    # -----Hyperparameters for threshold for binary classification task
    threshold = 0.5
    # -----Hyperparameters for information showing-----
    name_probability_output = "prob"
    name_classification_output = "class"
    name_regression_output = "pred"
    value_error_warning_task = "Argument of model function <task>: \"{}\" is not supported. It must be in " \
                               "[\"binary\", \"multi\", \"regression\"]".format(task)
    value_error_warning_optimizer = "Argument value of <optimizer>: {} is not supported.".format(optimizer)
    value_error_warning_output_size_and_task = "Argument of model function <output_size>: {}, must be 1 when <task> " \
                                               "is: \"{}\"".format(output_size, task)

    # ----------Assert for hyperparameters----------
    if task == "binary":
        assert (output_size == 1), value_error_warning_output_size_and_task
    if task == "regression":
        assert (output_size == 1), value_error_warning_output_size_and_task

    if seed != None:
        tf.set_random_seed(seed=seed)

    # ----------Build model inference----------
    with tf.variable_scope(name_or_scope="inference", reuse=reuse):
        with tf.name_scope(name="input-layer"):
            ids = features[name_feat_inds] # A tensor in shape of (None, field_size)
            x = features[name_feat_vals] # A tensor in shape of (None, field_size)

        with tf.name_scope(name="linear"):
            W = tf.get_variable(name="W",
                                shape=[feat_size, output_size],
                                dtype=dtype,
                                initializer=xavier_initializer(uniform=True, seed=seed, dtype=dtype),
                                regularizer=None)
            w = tf.nn.embedding_lookup(params=W, ids=ids) # A tensor in shape of (None, field_size, output_size)
            xreshape = tf.expand_dims(input=x, axis=-1) # A tensor in shape of (None, field_size, 1)
            z1 = tf.reduce_sum(input_tensor=tf.multiply(x=xreshape, y=w), # A tensor in shape of (None, field_size, output_size)
                               axis=1,
                               keepdims=False) # A tensor in shape of (None, output_size)

        with tf.name_scope(name="global-bias"):
            b = tf.get_variable(name="b",
                                shape=[output_size],
                                dtype=dtype,
                                initializer=tf.zeros_initializer(dtype=dtype),
                                regularizer=None,
                                trainable=use_global_bias)

        with tf.name_scope(name="output"):
            if use_global_bias:
                logits = z1 + b # A tensor in shape of (None, output_size)
            else:
                logits = z1 # A tensor in shape of (None, output_size)

            if task == "binary":
                logits = tf.squeeze(input=logits, axis=1) # A tensor in shape of (None)
                predictions = {
                    name_probability_output: tf.nn.sigmoid(x=logits),
                    name_classification_output: tf.cast(x=tf.greater(x=tf.nn.sigmoid(x=logits), y=threshold), dtype=tf.uint8)
                }
            elif task == "multi":
                predictions = {
                    name_probability_output: tf.nn.softmax(logits=logits, axis=-1),
                    name_classification_output: tf.argmax(input=logits, axis=-1, output_type=tf.int32)
                }
            elif task == "regression":
                logits = tf.squeeze(input=logits, axis=1) # A tensor in shape of (None)
                predictions = {
                    name_regression_output: logits
                }
            else:
                raise ValueError(value_error_warning_task)

    # ----------Provide an estimator spec for `ModeKeys.PREDICTION` mode----------
    if mode == tf.estimator.ModeKeys.PREDICT:
        export_outputs = {
            tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: tf.estimator.export.PredictOutput(outputs=predictions)
        } # For usage of tensorflow serving
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions, export_outputs=export_outputs)



    return predictions


if __name__ == '__main__':
    dtype = tf.float64
    features, labels = input_fn(filenames=["..//..//examples//Adventure-Works-Cycles//train.txt"],
                                delimiter=" ",
                                separator=":",
                                batch_size=32,
                                epochs=10,
                                shuffle=False,
                                dtype=dtype)
    hparams = {
        "task": "binary", # multi, binary, regression
        "output_size": 1, # 3, 1, 1
        "field_size": 13,
        "feat_size": 110,
        "use_global_bias": True,
        "lamb": 0.001,
        "optimizer": "Adam",
        "learning_rate": 0.0015,
        "dtype": dtype,
        "name_feat_inds": "inds",
        "name_feat_vals": "vals",
        "reuse": tf.AUTO_REUSE,
        "seed": 2019
    }

    result = model_fn(features=features, labels=labels, mode=tf.estimator.ModeKeys.TRAIN, params=hparams)
    with tf.Session() as sess:
        sess.run(fetches=tf.global_variables_initializer())
        r = sess.run(fetches=result)