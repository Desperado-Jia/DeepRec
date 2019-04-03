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
    """

    Parameters
    ----------
    :param filenames: list
        A list of string, containing one or more paths of filenames.
    :param delimiter: str
        A str, separating consecutive columns in data files.
    :param field_size_numerical: int
        
    :param batch_size: int

    :param epochs: int

    :param shuffle: bool
    :param buffer_size: int
    :param num_parallel_calls: int
    :param dtype:
    :param name_feat_vals_numerical:
    :param name_feat_inds_categorical:
    :return:
    """
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
        batch(batch_size=batch_size, drop_remainder=False)
    dataset = dataset.prefetch(buffer_size=1)
    iterator = dataset.make_one_shot_iterator()
    batch_feats, batch_labels = iterator.get_next()
    return batch_feats, batch_labels


def model_fn(features, labels, mode, params):
    # ----------Declare all hyperparameters from params----------
    task = params["task"]
    output_size = params["output_size"]
    field_size_numerical = params["field_size_numerical"]
    field_size_categorical = params["field_size_categorical"]
    feat_size_categorical = params["feat_size_categorical"]
    embed_size = params["embed_size"]
    num_cross_hidden_layers = params["num_cross_hidden_layers"]
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
    # -----Hyperparameters for exponential decay(*manual optional*)-----
    decay_steps = 5000
    decay_rate = 0.998
    staircase = True
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

    with tf.variable_scope(name_or_scope="inference", reuse=reuse):
        with tf.name_scope(name="inputs"):
            valsn = features[name_feat_vals_numerical]
            indsc = features[name_feat_inds_categorical] # A tensor in shape of (None, field_size_categorical)
            batch = tf.shape(input=valsn)[0]
            dim = field_size_numerical + field_size_categorical * embed_size

        with tf.name_scope(name="embed-and-stack-layer"):
            V = tf.get_variable(name="V",
                                shape=[feat_size_categorical, embed_size],
                                dtype=dtype,
                                initializer=xavier_initializer(uniform=False, seed=seed, dtype=dtype),
                                regularizer=None)
            embed = tf.nn.embedding_lookup(params=V, ids=indsc) # A tensor in shape of (None, field_size_categorical, embed_size)
            embed = tf.reshape(tensor=embed, shape=[-1, field_size_categorical * embed_size])
            x = tf.reshape(tensor=tf.concat(values=[valsn, embed], axis=-1), shape=[batch, dim])

        with tf.name_scope(name="deep-part"):
            ydeep = x
            for l in range(len(deep_hidden_sizes)):
                # -----The order for each hidden layer is: matmul => bn => relu => dropout => matmul => ...
                ydeep = tf.layers.dense(inputs=ydeep,
                                        units=deep_hidden_sizes[l],
                                        activation=None,
                                        use_bias=use_deep_hidden_bias,
                                        kernel_initializer=xavier_initializer(uniform=True, seed=seed, dtype=dtype),
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

        with tf.name_scope(name="cross-part"):
            Wc = tf.get_variable(name="Wc",
                                 shape=[num_cross_hidden_layers, dim],
                                 dtype=dtype,
                                 initializer=xavier_initializer(uniform=False, seed=seed, dtype=dtype),
                                 regularizer=l2_regularizer(scale=lamb))
            bc = tf.get_variable(name="bc",
                                 shape=[num_cross_hidden_layers, dim],
                                 dtype=dtype,
                                 initializer=xavier_initializer(uniform=False, seed=seed, dtype=dtype),
                                 regularizer=l2_regularizer(scale=lamb))

            ycross = x # A tensor in shape of (batch, dim)
            for l in range(num_cross_hidden_layers):
                wl = tf.expand_dims(input=Wc[l], axis=1) # A tensor in shape of (dim, 1)
                bl = bc[l] # A tensor in shape of (dim)
                xwl = tf.matmul(a=ycross, b=wl) # A tensor in shape of (batch, 1)
                ycross = tf.multiply(x=x, y=xwl) + ycross + bl


        with tf.name_scope(name="combine-output"):
            y = tf.concat(values=[ycross, ydeep], axis=-1) # A tensor in shape of (batch, concat_size)
            logits = tf.layers.dense(inputs=y,
                                     units=output_size,
                                     activation=None,
                                     use_bias=use_global_bias,
                                     kernel_initializer=xavier_initializer(uniform=True, seed=seed, dtype=dtype),
                                     bias_initializer=tf.zeros_initializer(dtype=dtype),
                                     kernel_regularizer=l2_regularizer(scale=lamb),
                                     bias_regularizer=None,
                                     name="output") # A tensor in shape of (batch, output_size)
            if task == "binary":
                logits = tf.squeeze(input=logits, axis=1) # A tensor in shape of (batch)
                predictions = {
                    name_probability_output: tf.sigmoid(x=logits),
                    name_classification_output: tf.cast(x=tf.round(x=tf.sigmoid(x=logits)), dtype=tf.uint8)
                }
            elif task == "regression":
                logits = tf.squeeze(input=logits, axis=1) # A tensor in shape of (batch)
                predictions = {
                    name_regression_output: logits
                }
            elif task == "multi":
                predictions = {
                    name_probability_output: tf.nn.softmax(logits=logits, axis=-1),
                    name_classification_output: tf.argmax(input=logits, axis=-1, output_type=tf.int32)
                }
            else:
                raise ValueError(value_error_warning_task)

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
    elif task == "multi":
        labels_one_hot = tf.one_hot(indices=tf.cast(x=labels, dtype=tf.int32),
                                    depth=output_size,
                                    axis=-1,
                                    dtype=dtype) # A tensor in shape of (None, output_size)
        loss = tf.reduce_mean(input_tensor=tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels_one_hot, logits=logits),
                              axis=0,
                              keepdims=False) # A scalar, representing the training loss of current batch training dataset
    else:
        raise ValueError(value_error_warning_task)

    reg = tf.reduce_sum(input_tensor=tf.get_collection(key=tf.GraphKeys.REGULARIZATION_LOSSES),
                        axis=0,
                        keepdims=False,
                        name="regularization") # A scalar, representing the regularization loss of current batch training dataset
    loss += reg

    # ----------Provide an estimator spec for `ModeKeys.EVAL` mode----------
    if mode == tf.estimator.ModeKeys.EVAL:
        if task == "binary":
            eval_metric_ops = {
                "accuracy": tf.metrics.accuracy(labels=labels, predictions=predictions[name_classification_output]),
                "precision": tf.metrics.precision(labels=labels, predictions=predictions[name_classification_output]),
                "recall": tf.metrics.recall(labels=labels, predictions=predictions[name_classification_output]),
                "auc": tf.metrics.auc(labels=labels, predictions=predictions[name_classification_output])
            }
        elif task == "multi":
            eval_metric_ops = {
                "recall": tf.metrics.recall(labels=labels, predictions=predictions[name_classification_output]) # ???
            }
        elif task == "regression":
            eval_metric_ops = {
                "rmse": tf.metrics.root_mean_squared_error(labels=labels, predictions=predictions[name_regression_output])
            }
        else:
            raise ValueError(value_error_warning_task)
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, predictions=predictions, eval_metric_ops=eval_metric_ops)

    # ----------Build optimizer----------
    global_step = tf.train.get_or_create_global_step(graph=tf.get_default_graph()) # Define a global step for training step counter
    if optimizer == "sgd":
        opt_op = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    elif optimizer == "sgd-with-exp-decay":
        decay_learning_rate = tf.train.exponential_decay(learning_rate=learning_rate,
                                                         global_step=global_step,
                                                         decay_steps=decay_steps,
                                                         decay_rate=decay_rate,
                                                         staircase=staircase)
        opt_op = tf.train.GradientDescentOptimizer(learning_rate=decay_learning_rate)
    elif optimizer == "momentum":
        opt_op = tf.train.MomentumOptimizer(learning_rate=learning_rate,
                                            momentum=0.9,
                                            use_nesterov=False)
    elif optimizer == "momentum-with-exp-decay":
        decay_learning_rate = tf.train.exponential_decay(learning_rate=learning_rate,
                                                         global_step=global_step,
                                                         decay_steps=decay_steps,
                                                         decay_rate=decay_rate,
                                                         staircase=staircase)
        opt_op = tf.train.MomentumOptimizer(learning_rate=decay_learning_rate,
                                            momentum=0.9,
                                            use_nesterov=False)
    elif optimizer == "nesterov":
        opt_op = tf.train.MomentumOptimizer(learning_rate=learning_rate,
                                            momentum=0.9,
                                            use_nesterov=True)
    elif optimizer == "nesterov-with-exp-decay":
        decay_learning_rate = tf.train.exponential_decay(learning_rate=learning_rate,
                                                         global_step=global_step,
                                                         decay_steps=decay_steps,
                                                         decay_rate=decay_rate,
                                                         staircase=staircase)
        opt_op = tf.train.MomentumOptimizer(learning_rate=decay_learning_rate,
                                            momentum=0.9,
                                            use_nesterov=True)
    elif optimizer == "adagrad":
        opt_op = tf.train.AdagradOptimizer(learning_rate=learning_rate,
                                           initial_accumulator_value=0.1)
    elif optimizer == "adadelta":
        opt_op = tf.train.AdadeltaOptimizer(learning_rate=learning_rate,
                                            rho=0.95)
    elif optimizer == "rmsprop":
        opt_op = tf.train.RMSPropOptimizer(learning_rate=learning_rate,
                                           decay=0.9)
    elif optimizer == "adam":
        opt_op = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                        beta1=0.9,
                                        beta2=0.999)
    else:
        raise NotImplementedError(value_error_warning_optimizer)

    train_op = opt_op.minimize(loss=loss, global_step=global_step, name="train_op")

    # ----------Provide an estimator spec for `ModeKeys.TRAIN` mode----------
    if (mode == tf.estimator.ModeKeys.TRAIN):
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)


def main(unused_argv):
    pass


if __name__ == '__main__':
    field_size_numerical = 2

    features, labels = input_fn(filenames=["data.txt"],
                                delimiter=" ",
                                field_size_numerical=field_size_numerical,
                                batch_size=32,
                                epochs=1,
                                shuffle=False)
    hparams = {
        "task": "multi",
        "output_size": 2,
        "field_size_numerical": field_size_numerical,
        "field_size_categorical": 6,
        "feat_size_categorical": 300,
        "embed_size": 16,
        "num_cross_hidden_layers": 5,
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