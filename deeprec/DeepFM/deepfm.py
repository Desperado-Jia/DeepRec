"""
Created on 2019/03/22 16:35
author: Tong Jia
email: cecilio.jia@gmail.com
description:
    An implementation of DeepFM: A Factorization-Machine based Neural Network for CTR Prediction (DeepFM).
    See algorithm and hyperparameter details:
        [Guo et al., 2017](https://www.ijcai.org/proceedings/2017/0239.pdf)
    The algorithm is developed with TensorFlow Estimator based on TensorFlow 1.12.0 version.
"""
import tensorflow as tf
import tensorflow.contrib as tfc
import os

FLAGS = tf.flags.FLAGS
# ----------Hyperparameters of estimator----------
tf.flags.DEFINE_enum(name="phase",
                     default=None,
                     enum_values=["train", "train-with-eval", "eval", "predict", "export"],
                     help="A string, representing the phase of estimator")
tf.flags.DEFINE_string(name="model_dir",
                       default=None,
                       help="A string, representing the folder path of saved model files")
tf.flags.DEFINE_string(name="export_dir",
                       default=None,
                       help="A string, representing the basic folder path of export .pb files")
# ----------Hyperparameters of estimator (optional)----------
tf.flags.DEFINE_integer(name="log_step_count_steps",
                        default=500,
                        help="(optional) An integer, representing the frequency, in number of global steps, that the global step/sec will be logged during training",
                        lower_bound=1,
                        upper_bound=None)
tf.flags.DEFINE_integer(name="save_checkpoints_steps",
                        default=20000,
                        help="(optional) An integer, representing save checkpoints every this many steps",
                        lower_bound=1,
                        upper_bound=None)
tf.flags.DEFINE_integer(name="keep_checkpoint_max",
                        default=2,
                        help="(optional) An integer, representing max number of saved model checkpoints",
                        lower_bound=1,
                        upper_bound=None)
# ----------Hyperparameters of input function----------
tf.flags.DEFINE_string(name="data_dir",
                       default=None,
                       help="A string, representing the folder path of dataset")
tf.flags.DEFINE_string(name="delimiter",
                       default=None,
                       help="A string, separating consecutive <index j>:<value j> pairs in a line of data file")
tf.flags.DEFINE_string(name="separator",
                       default=None,
                       help="A string, seprating feature index(left part) and feature value(right part) in a specific pair")
tf.flags.DEFINE_integer(name="batch_size",
                        default=None,
                        help="An integer, representing the number of consecutive elements of this dataset to combine in a single batch",
                        lower_bound=1,
                        upper_bound=None)
tf.flags.DEFINE_integer(name="epochs",
                        default=None,
                        help="An integer, representing the number of times the dataset should be repeated",
                        lower_bound=1,
                        upper_bound=None)
# ----------Hyperparameters of input function (optional)----------
tf.flags.DEFINE_boolean(name="shuffle",
                        default=True,
                        help="(optional) A boolean, instructing whether to randomly shuffle the samples of training dataset")
tf.flags.DEFINE_integer(name="buffer_size",
                        default=100000,
                        help="(optional) An integer scalar, denoting the number of bytes to buffer",
                        lower_bound=1,
                        upper_bound=None)
tf.flags.DEFINE_integer(name="num_parallel_calls",
                        default=4,
                        help="(optional) An integer scalar, representing the number elements to process in parallel",
                        lower_bound=1,
                        upper_bound=None)
tf.flags.DEFINE_boolean(name="use_dtype_high_precision",
                        default=False,
                        help="(optional) A boolean, instructing the dtype of both input function and model function. If False, use tf.float32; if True, use tf.float64")
tf.flags.DEFINE_string(name="name_feat_inds",
                       default="inds",
                       help="(optional) A string, representing the name of feature indices in return dict")
tf.flags.DEFINE_string(name="name_feat_vals",
                       default="vals",
                       help="(optional) A string, representing the name of feature values in return dict")
# ----------Hyperparameters of model function----------
tf.flags.DEFINE_string(name="task",
                       default=None,
                       help="A string, representing the type of task (binary or regression)")
tf.flags.DEFINE_integer(name="field_size",
                        default=None,
                        help="An integer scalar, representing the number of fields(number of columns before one-hot encoding) of dataset")
tf.flags.DEFINE_integer(name="feat_size",
                        default=None,
                        help="An integer scalar, representing the number of features(number of columns after one-hot encoding) of dataset")
tf.flags.DEFINE_integer(name="embed_size",
                        default=None,
                        help="An integer scalar, representing the dimension of embedding vectors for all features")
tf.flags.DEFINE_list(name="hidden_sizes",
                     default=None,
                     help="A list, containing the number of hidden units of each hidden layer in dnn part")
tf.flags.DEFINE_list(name="dropouts",
                     default=None,
                     help="A list, containing the dropout rate of each hidden layer in dnn part (e.g. 128,64,64,32); if None, don't use dropout operation for any hidden layer")
# ----------Hyperparameters of model function (optional)----------
tf.flags.DEFINE_boolean(name="use_global_bias",
                        default=True,
                        help="(optional) A boolean, instructing whether to use global bias in model inference")
tf.flags.DEFINE_boolean(name="use_linear",
                        default=True,
                        help="(optional) A boolean, instructing whether to use linear part in model inference")
tf.flags.DEFINE_boolean(name="use_hidden_bias",
                        default=True,
                        help="(optional) A boolean, instructing whether to use bias of hidden layer units in model inference")
tf.flags.DEFINE_boolean(name="use_bn",
                        default=True,
                        help="(optional) A boolean, instructing whether to use batch normalization for each hidden layer in dnn part")
tf.flags.DEFINE_float(name="lamb",
                      default=0.001,
                      help="(optional) A float scalar, representing the coefficient of regularization term",
                      lower_bound=0.0,
                      upper_bound=None)
tf.flags.DEFINE_string(name="optimizer",
                       default="adam",
                       help="(optional) A string, representing the type of optimizer")
tf.flags.DEFINE_float(name="learning_rate",
                      default=0.003,
                      help="(optional) A float scalar, representing the learning rate of optimizer",
                      lower_bound=0.0,
                      upper_bound=None)

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
    """Model function of DeepFM for predictive analytics of high dimensional sparse data.

    Args of dict params:
        task: str
            A string, representing the type of task.
            Note:
                it must take value from ["binary", "regression"];
                it instruct the type of loss function:
                    "binary": sigmoid cross-entropy;
                    "regression": mean squared error.
        field_size: int
            An integer scalar, representing the number of fields(number of columns before one-hot encoding) of dataset.
        feat_size: int
            An integer scalar, representing the number of features(number of columns after one-hot encoding) of dataset.
        embed_size: int
            An integer scalar, representing the dimension of embedding vectors for all features.
        hidden_sizes: list
            A list, containing the number of hidden units of each hidden layer in dnn part.
            Note:
                it doesn't contain output layer of dnn part.
        dropouts: list or None
            If list, containing the dropout rate of each hidden layer in dnn part;
            If None, don't use dropout operation for any hidden layer.
            Note:
                if list, the length of <dropouts> must be equal to <hidden_sizes>.
        use_global_bias: bool
            A boolean, instructing whether to use global bias in model inference.
        use_linear: bool
            A boolean, instructing whether to use linear part in model inference.
        use_hidden_bias: bool
            A boolean, instructing whether to use bias of hidden layer units in model inference.
        use_bn: bool
            A boolean, instructing whether to use batch normalization for each hidden layer in dnn part.
        lamb: float
            A float scalar, representing the coefficient of regularization term (the larger the value of lamb, the stronger the penalty is).
        optimizer: str
            A string, representing the type of optimizer.
        learning_rate: float
            A float scalar, representing the learning rate of optimizer.
        dtype: tf.Dtype
            A tf.DType, representing the numeric type of values.
            Note:
                it must take value from [tf.float32, tf.float64];
                it must be consistent with <dtype> of input function.
        name_feat_inds: str
            A string, representing the name of feature indices in return dict.
            Note:
                it must be consistent with <name_feat_inds> of input function.
        name_feat_vals: str
            A string, representing the name of feature values in return dict.
            Note:
                it must be consistent with <name_feat_vals> of input function.
        reuse: bool
            A boolean, which takes value from [False, True, tf.AUTO_REUSE].
        seed: int or None
            If integer scalar, representing the random seed of tensorflow;
            If None, random choice.
    """
    # ----------Declare all hyperparameters from params----------
    task = params["task"]
    field_size = params["field_size"]
    feat_size = params["feat_size"]
    embed_size = params["embed_size"]
    hidden_sizes = params["hidden_sizes"]
    dropouts = params["dropouts"]
    use_global_bias = params["use_global_bias"]
    use_linear = params["use_linear"]
    use_hidden_bias = params["use_hidden_bias"]
    use_bn = params["use_bn"]
    lamb = params["lamb"]
    optimizer = params["optimizer"]
    learning_rate = params["learning_rate"]
    dtype = params["dtype"]
    name_feat_inds = params["name_feat_inds"]
    name_feat_vals = params["name_feat_vals"]
    reuse = params["reuse"]
    seed = params["seed"]

    if seed != None:
        tf.set_random_seed(seed=seed)

    NAME_PROBABILITY_OUTPUT = "prob"
    NAME_CLASSIFICATION_OUTPUT = "class"
    NAME_REGRESSION_OUTPUT = "pred"

    VALUE_ERROR_WARNING_TASK = "Argument of model function <task>: \"{}\" is invalid. It must be in [\"binary\", \"regression\"]".format(task)
    VALUE_ERROR_WARNING_OPTIMIZER = "Argument value of <optimizer>: {} is not supported.".format(optimizer)

    # ----------Hyperparameters for exponential decay(*manual optional*)----------
    DECAY_STEPS = 5000
    DECAY_RATE = 0.998
    STAIRCASE = True

    # ----------Assert for hyperparameters----------
    assert (task in ["binary", "regression"])
    if dropouts != None:
        assert (len(dropouts) == len(hidden_sizes))
    assert (dtype in [tf.float32, tf.float64])

    # ----------Build model inference----------
    with tf.variable_scope(name_or_scope="inference", reuse=reuse):
        with tf.name_scope(name="inputs"):
            ids = features[name_feat_inds]
            x = features[name_feat_vals]

        with tf.name_scope(name="global-bias"):
            b = tf.get_variable(name="b",
                                shape=[1],
                                dtype=dtype,
                                initializer=tf.zeros_initializer(dtype=dtype),
                                regularizer=tfc.layers.l2_regularizer(scale=lamb),
                                trainable=use_global_bias)

        with tf.name_scope(name="lr-part"):
            W = tf.get_variable(name="W",
                                shape=[feat_size],
                                dtype=dtype,
                                initializer=tfc.layers.xavier_initializer(uniform=True, dtype=dtype), # *manual optional*
                                regularizer=tfc.layers.l2_regularizer(scale=lamb),
                                trainable=use_linear)
            # -----embedding lookup op for first order weights-----
            w = tf.nn.embedding_lookup(params=W, ids=ids) # A tensor in shape of (None, field_size)
            ylr = tf.reduce_sum(input_tensor=tf.multiply(x=x, y=w), # A tensor in shape of (None, field_size)
                                axis=1,
                                keepdims=False) # A tensor in shape of (None)

        with tf.name_scope(name="fm-part"):
            V = tf.get_variable(name="V",
                                shape=[feat_size, embed_size],
                                dtype=dtype,
                                initializer=tfc.layers.xavier_initializer(uniform=False, dtype=dtype), # manual optional
                                regularizer=None, # *manual optional*
                                trainable=True)
            # -----embedding lookup op for second order weights-----
            v = tf.nn.embedding_lookup(params=V, ids=ids) # A tensor in shape of (None, field_size, embed_size)
            xreshape = tf.expand_dims(input=x, axis=-1) # A tensor in shape of (None, field_size, 1)
            vx = tf.multiply(x=xreshape, y=v) # A tensor in shape of (None, field_size, embed_size)
            p = tf.square(x=tf.reduce_sum(input_tensor=vx, axis=1, keepdims=False)) # a tensor in shape of (None, embedding_size)
            q = tf.reduce_sum(input_tensor=tf.square(x=vx), # A tensor in shape of (None, field_size, embedding_size)
                              axis=1,
                              keepdims=False) # A tensor in shape of (None, embedding_size)
            yfm = 0.5 * tf.reduce_sum(input_tensor=tf.subtract(x=p, y=q), # A tensor in shape of (None, embedding_size)
                                      axis=1,
                                      keepdims=False) # A tensor in shape of (None)

        with tf.name_scope(name="mlp-part"):
            ymlp = tf.reshape(tensor=vx, shape=[-1, field_size * embed_size]) # a tensor in shape of (None, field_size * embed_size)
            for l in range(len(hidden_sizes)):
                # -----The order for each hidden layer is: matmul => bn => relu => dropout => matmul => ...
                ymlp = tf.layers.dense(inputs=ymlp,
                                       units=hidden_sizes[l],
                                       activation=None,
                                       use_bias=use_hidden_bias,
                                       kernel_initializer=tfc.layers.xavier_initializer(uniform=True, dtype=dtype),
                                       bias_initializer=tf.zeros_initializer(dtype=dtype),
                                       kernel_regularizer=None, # *manual optional*
                                       bias_regularizer=None, # *manual optional*
                                       trainable=True,
                                       name="mlp-dense-hidden-{}".format(l))
                if use_bn == True:
                    ymlp = tf.layers.batch_normalization(inputs=ymlp,
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
                                                         trainable=True,
                                                         name="mlp-bn-hidden-{}".format(l))

                ymlp = tf.nn.relu(features=ymlp)
                if dropouts != None:
                    ymlp = tf.layers.dropout(inputs=ymlp,
                                             rate=dropouts[l],
                                             seed=seed,
                                             training=(mode == tf.estimator.ModeKeys.TRAIN))
            # -----output layer of mlp part-----
            ymlp = tf.layers.dense(inputs=ymlp,
                                   units=1,
                                   activation=None,
                                   use_bias=False,
                                   kernel_initializer=tfc.layers.xavier_initializer(uniform=True, dtype=dtype),
                                   kernel_regularizer=None, # *manual optional*
                                   trainable=True,
                                   name="mlp-dense-output") # A tensor in shape of (None, 1)
            ymlp = tf.squeeze(input=ymlp) # A tensor in shape of (None)

        with tf.name_scope(name="output"):
            if use_global_bias == False and use_linear == False:
                logits = yfm + ymlp
            elif use_global_bias == False and use_linear == True:
                logits = ylr + yfm + ymlp
            elif use_global_bias == True and use_linear == False:
                logits = b + yfm + ymlp
            else:
                logits = b + ylr + yfm + ymlp

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

    # ----------Provide an estimator spec for `ModeKeys.EVAL` mode----------
    if mode == tf.estimator.ModeKeys.EVAL:
        if task == "binary":
            eval_metric_ops = {
                "accuracy": tf.metrics.accuracy(labels=labels, predictions=predictions[NAME_CLASSIFICATION_OUTPUT]),
                "precision": tf.metrics.precision(labels=labels, predictions=predictions[NAME_CLASSIFICATION_OUTPUT]),
                "recall": tf.metrics.recall(labels=labels, predictions=predictions[NAME_CLASSIFICATION_OUTPUT]),
                "auc": tf.metrics.auc(labels=labels, predictions=predictions[NAME_CLASSIFICATION_OUTPUT])
            }
        elif task == "regression":
            eval_metric_ops = {
                "rmse": tf.metrics.root_mean_squared_error(labels=labels, predictions=predictions[NAME_REGRESSION_OUTPUT])
            }
        else:
            raise ValueError(VALUE_ERROR_WARNING_TASK)
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, predictions=predictions, eval_metric_ops=eval_metric_ops)

    # ----------Build optimizer----------
    global_step = tf.train.get_or_create_global_step(graph=tf.get_default_graph()) # Define a global step for training step counter
    if optimizer == "sgd":
        opt_op = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    elif optimizer == "sgd-with-exp-decay":
        decay_learning_rate = tf.train.exponential_decay(learning_rate=learning_rate,
                                                         global_step=global_step,
                                                         decay_steps=DECAY_STEPS,
                                                         decay_rate=DECAY_RATE,
                                                         staircase=STAIRCASE)
        opt_op = tf.train.GradientDescentOptimizer(learning_rate=decay_learning_rate)
    elif optimizer == "momentum":
        opt_op = tf.train.MomentumOptimizer(learning_rate=learning_rate,
                                            momentum=0.9,
                                            use_nesterov=False)
    elif optimizer == "momentum-with-exp-decay":
        decay_learning_rate = tf.train.exponential_decay(learning_rate=learning_rate,
                                                         global_step=global_step,
                                                         decay_steps=DECAY_STEPS,
                                                         decay_rate=DECAY_RATE,
                                                         staircase=STAIRCASE)
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
                                                         decay_steps=DECAY_STEPS,
                                                         decay_rate=DECAY_RATE,
                                                         staircase=STAIRCASE)
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
        raise NotImplementedError(VALUE_ERROR_WARNING_OPTIMIZER)

    train_op = opt_op.minimize(loss=loss, global_step=global_step, name="train_op")

    # ----------Provide an estimator spec for `ModeKeys.TRAIN` mode----------
    if (mode == tf.estimator.ModeKeys.TRAIN):
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)


def main(unused_argv):
    # ----------Declare all hyperparameters----------
    phase = FLAGS.phase
    log_step_count_steps = FLAGS.log_step_count_steps # (optional)
    save_checkpoints_steps = FLAGS.save_checkpoints_steps # (optional)
    keep_checkpoint_max = FLAGS.keep_checkpoint_max # (optional)
    model_dir = FLAGS.model_dir
    export_dir = FLAGS.export_dir
    data_dir = FLAGS.data_dir
    delimiter = FLAGS.delimiter
    separator = FLAGS.separator
    batch_size = FLAGS.batch_size
    epochs = FLAGS.epochs
    shuffle = FLAGS.shuffle
    buffer_size = FLAGS.buffer_size # (optional)
    num_parallel_calls = FLAGS.num_parallel_calls # (optional)
    use_dtype_high_precision = FLAGS.use_dtype_high_precision # (optional)
    name_feat_inds = FLAGS.name_feat_inds # (optional)
    name_feat_vals = FLAGS.name_feat_vals # (optional)
    task = FLAGS.task
    field_size = FLAGS.field_size
    feat_size = FLAGS.feat_size
    embed_size = FLAGS.embed_size
    hidden_sizes = FLAGS.hidden_sizes
    dropouts = FLAGS.dropouts
    use_global_bias = FLAGS.use_global_bias # (optional)
    use_linear = FLAGS.use_linear # (optional)
    use_hidden_bias = FLAGS.use_hidden_bias # (optional)
    use_bn = FLAGS.use_bn # (optional)
    lamb = FLAGS.lamb
    optimizer = FLAGS.optimizer
    learning_rate = FLAGS.learning_rate

    PREFIX_TRAIN_FILE = "train"
    PREFIX_EVAL_FILE = "eval"
    PREFIX_PREDICT_FILE = "predict"
    REUSE = False
    SEED = None

    if use_dtype_high_precision == False:
        dtype = tf.float32
    else:
        dtype = tf.float64
    hidden_sizes = [int(float(ele)) for ele in hidden_sizes]
    if dropouts != None:
        dropouts = [float(ele) for ele in dropouts]

    hparams = {
        "task": task,
        "field_size": field_size,
        "feat_size": feat_size,
        "embed_size": embed_size,
        "hidden_sizes": hidden_sizes,
        "dropouts": dropouts,
        "use_global_bias": use_global_bias,
        "use_linear": use_linear,
        "use_hidden_bias": use_hidden_bias,
        "use_bn": use_bn,
        "lamb": lamb,
        "optimizer": optimizer,
        "learning_rate": learning_rate,
        "dtype": dtype,
        "name_feat_inds": name_feat_inds,
        "name_feat_vals": name_feat_vals,
        "reuse": REUSE,
        "seed": SEED
    }
    config = tf.estimator.RunConfig(
        log_step_count_steps=log_step_count_steps,
        save_checkpoints_steps=save_checkpoints_steps,
        keep_checkpoint_max=keep_checkpoint_max
    )
    estimator = tf.estimator.Estimator(model_fn=model_fn, model_dir=model_dir, params=hparams, config=config)
    if phase == "train":
        filenames_train = tf.gfile.Glob(filename=os.path.join(data_dir, PREFIX_TRAIN_FILE + "*"))
        estimator.train(
            input_fn=lambda : input_fn(
                filenames=filenames_train,
                delimiter=delimiter,
                separator=separator,
                batch_size=batch_size,
                epochs=epochs,
                shuffle=shuffle,
                buffer_size=buffer_size,
                num_parallel_calls=num_parallel_calls,
                dtype=dtype,
                name_feat_inds=name_feat_inds,
                name_feat_vals=name_feat_vals
            ),
            steps=None,
            max_steps=None
        )
    elif phase == "train-with-eval":
        filenames_train = tf.gfile.Glob(filename=os.path.join(data_dir, PREFIX_TRAIN_FILE + "*"))
        filenames_eval = tf.gfile.Glob(filename=os.path.join(data_dir, PREFIX_EVAL_FILE + "*"))
        train_spec = tf.estimator.TrainSpec(
            input_fn=lambda : input_fn(
                filenames=filenames_train,
                delimiter=delimiter,
                separator=separator,
                batch_size=batch_size,
                epochs=epochs,
                shuffle=shuffle,
                buffer_size=buffer_size,
                num_parallel_calls=num_parallel_calls,
                dtype=dtype,
                name_feat_inds=name_feat_inds,
                name_feat_vals=name_feat_vals
            ),
            max_steps=None
        )
        eval_spec = tf.estimator.EvalSpec(
            input_fn=lambda : input_fn(
                filenames=filenames_eval,
                delimiter=delimiter,
                separator=separator,
                batch_size=batch_size,
                epochs=1,
                shuffle=False,
                buffer_size=buffer_size,
                num_parallel_calls=num_parallel_calls,
                dtype=dtype,
                name_feat_inds=name_feat_inds,
                name_feat_vals=name_feat_vals
            ),
            steps=None,
            start_delay_secs=120,
            throttle_secs=600
        )
        tf.estimator.train_and_evaluate(estimator=estimator, train_spec=train_spec, eval_spec=eval_spec)
    elif phase == "eval":
        filenames_eval = tf.gfile.Glob(filename=os.path.join(data_dir, PREFIX_EVAL_FILE + "*"))
        estimator.evaluate(
            input_fn=lambda : input_fn(
                filenames=filenames_eval,
                delimiter=delimiter,
                separator=separator,
                batch_size=batch_size,
                epochs=1,
                shuffle=False,
                buffer_size=buffer_size,
                num_parallel_calls=num_parallel_calls,
                dtype=dtype,
                name_feat_inds=name_feat_inds,
                name_feat_vals=name_feat_vals
            )
        )
    elif phase == "predict":
        filenames_predict = tf.gfile.Glob(filename=os.path.join(data_dir, PREFIX_PREDICT_FILE + "*"))
        P = estimator.predict(
            input_fn=lambda : input_fn(
                filenames=filenames_predict,
                delimiter=delimiter,
                separator=separator,
                batch_size=batch_size,
                epochs=1,
                shuffle=False,
                buffer_size=buffer_size,
                num_parallel_calls=num_parallel_calls,
                dtype=dtype,
                name_feat_inds=name_feat_inds,
                name_feat_vals=name_feat_vals
            )
        )
        for p in P:
            print(p)
    elif phase == "export":
        features = {
            name_feat_inds: tf.placeholder(dtype=tf.int32, shape=[None, field_size], name=name_feat_inds),
            name_feat_vals: tf.placeholder(dtype=dtype, shape=[None, field_size], name=name_feat_vals)
        }
        serving_input_receiver_fn = tf.estimator.export.build_raw_serving_input_receiver_fn(features=features)
        estimator.export_savedmodel(export_dir_base=export_dir, serving_input_receiver_fn=serving_input_receiver_fn)
    else:
        raise NotImplementedError("Argument <phase> value: {} is not supported.".format(phase))

if __name__ == '__main__':
    tf.logging.set_verbosity(v=tf.logging.INFO)
    tf.app.run(main=main)