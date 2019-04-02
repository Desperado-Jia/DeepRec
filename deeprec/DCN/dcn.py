"""
Created on 2019/04/02 15:56
author: Tong Jia
email: cecilio.jia@gmail.com
description:
    An implementation of Deep & Cross Network for Ad Click Predictions (DCN).
    See algorithm and hyperparameter details:
        [Wang et al., 2017](https://arxiv.org/pdf/1708.05123.pdf)
    The algorithm is developed with TensorFlow Estimator based on TensorFlow 1.12.0 version.
"""
import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer, l2_regularizer

def input_fn(filenames,
             delimiter,
             field_size_numerical,
             batch_size,
             epochs,
             shuffle=True,
             buffer_size=100000,
             num_parallel_calls=4,
             dtype=tf.float32,
             name_feat_vals_numerical="vals_numerical",
             name_feat_inds_categorical="inds_categorical"):
    def map_func(line):
        columns = tf.string_split(source=[line], delimiter=delimiter, skip_empty=False).values
        # ----------Process label----------
        label = tf.string_to_number(string_tensor=columns[0], out_type=dtype)
        # ----------Process numerical fields----------
        vals_numerical = tf.string_to_number(string_tensor=columns[1: 1 + field_size_numerical],
                                             out_type=dtype) # A tensor in shape of (field_size_numerical)
        # ----------Process categorical fields----------
        inds_categorical = tf.string_to_number(string_tensor=columns[1 + field_size_numerical: ],
                                               out_type=tf.int32) # A tensor in shape of (field_size_categorical)
        feats = {
            name_feat_vals_numerical: vals_numerical,
            name_feat_inds_categorical: inds_categorical
        }
        return feats, label

    dataset = tf.data.TextLineDataset(filenames=filenames, buffer_size=buffer_size).\
        map(map_func=map_func, num_parallel_calls=num_parallel_calls)
    if shuffle:
        dataset = dataset.shuffle(buffer_size=buffer_size)
    dataset = dataset. \
        repeat(count=epochs). \
        batch(batch_size=batch_size, drop_remainder=True)
    dataset = dataset.prefetch(buffer_size=1)
    iterator = dataset.make_one_shot_iterator()
    batch_feats, batch_labels = iterator.get_next()
    return batch_feats, batch_labels


def model_fn(features, labels, mode, params):
    # ----------Declare all hyperparameters from params----------
    task = params["task"]
    field_size_numerical = params["field_size_numerical"]
    field_size_categorical = params["field_size_categorical"]
    feat_size_categorical = params["feat_size_categorical"]
    embed_size = params["embed_size"]
    deep_hidden_sizes = params["deep_hidden_sizes"]
    dropouts = params["dropouts"]
    use_global_bias = params["use_global_bias"]
    use_deep_hidden_bias = params["use_deep_hidden_bias"]
    use_bn = params["use_bn"]
    lamb = params["lamb"]
    optimizer = params["optimizer"]
    learning_rate = params["learning_rate"]
    name_feat_vals_numerical = params["name_feat_vals_numerical"]
    name_feat_inds_categorical = params["name_feat_inds_categorical"]
    dtype = params["dtype"]
    reuse = params["reuse"]
    seed = params["seed"]

    if seed != None:
        tf.set_random_seed(seed=seed)

    with tf.variable_scope(name_or_scope="inference", reuse=reuse):
        with tf.name_scope(name="inputs"):
            valsn = features[name_feat_vals_numerical]
            indsc = features[name_feat_inds_categorical] # A tensor in shape of (None, field_size_categorical)

        with tf.name_scope(name="embed-and-stack-layer"):
            V = tf.get_variable(name="V",
                                shape=[feat_size_categorical, embed_size],
                                dtype=dtype,
                                initializer=xavier_initializer(uniform=False, dtype=dtype),
                                regularizer=None)
            embed = tf.nn.embedding_lookup(params=V, ids=indsc) # A tensor in shape of (None, field_size_categorical, embed_size)
            embed = tf.reshape(tensor=embed, shape=[-1, field_size_categorical * embed_size])
            x = tf.concat(values=[valsn, embed], axis=1)

        with tf.name_scope(name="deep-part"):
            # ydeep = tf.reshape(tensor=x, shape=[-1, field_size_numerical + field_size_categorical * embed_size])
            for l in range(len(deep_hidden_sizes)):
                # -----The order for each hidden layer is: matmul => bn => relu => dropout => matmul => ...
                ydeep = tf.layers.dense(inputs=ydeep,
                                        units=deep_hidden_sizes[l],
                                        activation=None,
                                        use_bias=use_deep_hidden_bias,
                                        kernel_initializer=xavier_initializer(uniform=True, dtype=dtype),
                                        bias_initializer=tf.zeros_initializer(dtype=dtype),
                                        kernel_regularizer=l2_regularizer(scale=lamb),
                                        bias_regularizer=None,
                                        name="deep-dense-hidden-{}".format(l))
                if use_bn == True:
                    ydeep = tf.layers.batch_normalization(inputs=ydeep,
                                                          axis=-1,
                                                          momentum=0.99, # *manual optional*
                                                          epsilon=1e-3, # *manual optional*
                                                          center=True,
                                                          scale=True,
                                                          beta_initializer=tf.zeros_initializer(dtype=dtype),
                                                          gamma_initializer=tf.ones_initializer(dtype=dtype),
                                                          moving_mean_initializer=tf.zeros_initializer(dtype=dtype),
                                                          moving_variance_initializer=tf.ones_initializer(dtype=dtype),
                                                          beta_regularizer=None, # *manual optional*
                                                          gamma_regularizer=None, # *manual optional*
                                                          training=(mode == tf.estimator.ModeKeys.TRAIN),
                                                          name="dense-bn-hidden-{}".format(l))
                ydeep = tf.nn.relu(features=ydeep)
                if dropouts != None:
                    ydeep = tf.layers.dropout(inputs=ydeep,
                                             rate=dropouts[l],
                                             seed=seed,
                                             training=(mode == tf.estimator.ModeKeys.TRAIN))

    return ydeep


if __name__ == '__main__':
    field_size_numerical = 2

    features, labels = input_fn(filenames=["data.txt"],
                                delimiter=" ",
                                field_size_numerical=field_size_numerical,
                                batch_size=2,
                                epochs=10,
                                shuffle=False)
    hparams = {
        "task": "binary",
        "field_size_numerical": field_size_numerical,
        "field_size_categorical": 5,
        "feat_size_categorical": 300,
        "embed_size": 16,
        "deep_hidden_sizes": [64, 64, 32, 32],
        "dropouts": None,
        "use_global_bias": True,
        "use_deep_hidden_bias": True,
        "use_bn": True,
        "lamb": 0.001,
        "optimizer": "adam",
        "learning_rate": 0.0025,
        "name_feat_vals_numerical": "vals_numerical",
        "name_feat_inds_categorical": "inds_categorical",
        "dtype": tf.float32,
        "reuse": tf.AUTO_REUSE,
        "seed": 2019
    }
    result = model_fn(features=features, labels=labels, mode=tf.estimator.ModeKeys.TRAIN, params=hparams)

    with tf.Session() as sess:
        sess.run(fetches=tf.global_variables_initializer())
        r = sess.run(fetches=result)