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
from tensorflow.contrib.layers import xavier_initializer, l2_regularizer

def input_fn(filenames,
             delimiter,
             separator,
             epochs,
             batch_size,
             shuffle=True,
             buffer_size=100000,
             num_parallel_calls=4,
             dtype=tf.float32,
             name_feat_inds="inds",
             name_feat_vals="vals"):
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
    dataset = dataset.repeat(count=epochs).batch(batch_size=batch_size, drop_remainder=False)
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
    use_global_bias = params["use_global_bias"]
    use_linear = params["use_linear"]
    lamb = params["lamb"]
    dtype = params["dtype"]
    name_feat_inds = params["name_feat_inds"]
    name_feat_vals = params["name_feat_vals"]
    reuse = params["reuse"]
    seed = params["seed"]

    if seed != None:
        tf.set_random_seed(seed=seed)

    # ----------Assert for hyperparameters----------
    NAME_PROBABILITY_OUTPUT = "prob"
    NAME_CLASSIFICATION_OUTPUT = "class"
    NAME_REGRESSION_OUTPUT = "pred"

    VALUE_ERROR_WARNING_TASK = "Argument of model function <task>: \"{}\" is not supported. It must be in [\"binary\", \"regression\"]".format(task)

    # ----------Build model inference----------
    with tf.variable_scope(name_or_scope="inference", reuse=reuse):
        with tf.name_scope(name="inputs"):
            ids = features[name_feat_inds] # A tensor in shape of (None, field_size)
            x = features[name_feat_vals] # A tensor in shape of (None, field_size)

        with tf.name_scope(name="global-bias"):
            b = tf.get_variable(name="b",
                                shape=[1],
                                dtype=dtype,
                                initializer=tf.zeros_initializer(dtype=dtype),
                                regularizer=l2_regularizer(scale=lamb),
                                trainable=use_global_bias)

        with tf.name_scope(name="lr-part"):
            W = tf.get_variable(name="W",
                                shape=[feat_size],
                                dtype=dtype,
                                initializer=xavier_initializer(uniform=True, dtype=dtype), # *manual optional*
                                regularizer=l2_regularizer(scale=lamb),
                                trainable=use_linear)
            # -----embedding lookup op for first order weights-----
            w = tf.nn.embedding_lookup(params=W, ids=ids) # A tensor in shape of (None, field_size)
            ylr = tf.reduce_sum(input_tensor=tf.multiply(x=x, y=w), # A tensor in shape of (None, field_size)
                                axis=1,
                                keepdims=False) # A tensor in shape of (None)

        with tf.name_scope(name="ffm-part"):
            V = tf.get_variable(name="V",
                                shape=[feat_size, field_size, embed_size],
                                dtype=dtype,
                                initializer=xavier_initializer(uniform=False, dtype=dtype),
                                regularizer=None)
            embedding = tf.nn.embedding_lookup(params=V, ids=ids) # A tensor in shape of (None, field_size, field_size, embed_size)
            yffm = tf.constant(value=1, dtype=dtype, shape=[1])
            for i in range(field_size):
                # -----for each valid feature
                for j in range(i + 1, field_size):
                    # -----for each field j to be interacted with feature i
                    key = embedding[:, i, j, :] # A tensor in shape of (None, embed_size)
                    query = embedding[: , j, i, :] # A tensor in shape of (None, embed_size)
                    weight = tf.multiply(x=key, y=query) # A tensor in shape of (None, embed_size)
                    xixj = tf.expand_dims(input=tf.multiply(x=x[:, i], y=x[: ,j]), axis=-1) # A tensor in shape of (None, 1)
                    value = tf.reduce_sum(input_tensor=tf.multiply(x=weight, y=xixj), # A tensor in shape of (None, embed_size)
                                          axis=-1,
                                          keepdims=False) # A tensor in shape of (None)
                    yffm += value

        with tf.name_scope(name="output"):
            if use_global_bias == False and use_linear == False:
                logits = yffm
            elif use_global_bias == False and use_linear == True:
                logits = ylr + yffm
            elif use_global_bias == True and use_linear == False:
                logits = b + yffm
            else:
                logits = b + ylr + yffm

            if task == "binary":
                predictions = {
                    NAME_PROBABILITY_OUTPUT: tf.sigmoid(x=logits),
                    NAME_CLASSIFICATION_OUTPUT: tf.cast(x=tf.round(x=tf.sigmoid(x=logits)), dtype=tf.uint8)
                }
            elif task == "regression":
                predictions = {
                    NAME_REGRESSION_OUTPUT: logits
                }
            else:
                raise ValueError(VALUE_ERROR_WARNING_TASK)

    # ----------Provide an estimator spec for `ModeKeys.PREDICTION` mode----------
    if mode == tf.estimator.ModeKeys.PREDICT:
        export_outputs = {
            tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: tf.estimator.export.PredictOutput(outputs=predictions)
        } # For usage of tensorflow serving
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions, export_outputs=export_outputs)

    # ----------Build loss function----------
    if task == "binary":
        loss = tf.reduce_mean(input_tensor=tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits),
                              axis=0,
                              keepdims=False) # A scalar, representing the training loss of current batch training dataset
    elif task == "regression":
        loss = tf.reduce_mean(input_tensor=tf.square(x=tf.subtract(x=labels, y=logits)),
                              axis=0,
                              keepdims=False) # A scalar, representing the training loss of current batch training dataset
    else:
        raise ValueError(VALUE_ERROR_WARNING_TASK)

    reg = tf.reduce_sum(input_tensor=tf.get_collection(key=tf.GraphKeys.REGULARIZATION_LOSSES),
                        axis=0,
                        keepdims=False,
                        name="regularization") # A scalar, representing the regularization loss of current batch training dataset
    loss += reg

    return loss


if __name__ == '__main__':
    features, labels = input_fn(filenames=["..//..//examples//Adventure-Works-Cycles//train.txt"],
                                delimiter=" ",
                                separator=":",
                                epochs=10,
                                batch_size=16,
                                shuffle=False)
    hparams = {
        "task": "binary",
        "field_size": 13,
        "feat_size": 110,
        "embed_size": 8,
        "use_global_bias": True,
        "use_linear": True,
        "lamb": 0.001,
        "dtype": tf.float32,
        "name_feat_inds": "inds",
        "name_feat_vals": "vals",
        "reuse": tf.AUTO_REUSE,
        "seed": 2019
    }
    result = model_fn(features=features, labels=labels, mode=tf.estimator.ModeKeys.TRAIN, params=hparams)

    with tf.Session() as sess:
        sess.run(fetches=tf.global_variables_initializer())
        r = sess.run(fetches=result)