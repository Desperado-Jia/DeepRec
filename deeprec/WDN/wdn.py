"""
Created on 2019/04/09 14:43
author: Tong Jia
email: cecilio.jia@gmail.com
description:
    An implement of Wide & Deep Learning for Recommender Systems (WDN).
    See algorithm and hyperparameter details:
        [Cheng et al., 2016](https://arxiv.org/pdf/1606.07792.pdf)
    The algorithm is developed with TensorFlow Estimator based on TensorFlow 1.12.0 version.
"""
import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer, l2_regularizer
import os

FLAGS = tf.flags.FLAGS
# ----------Hyperparameters of estimator----------
tf.flags.DEFINE_enum(name="phase",
                     default=None,
                     enum_values=["train", "eval", "predict", "export"],
                     help="A string, representing the phase of estimator")
tf.flags.DEFINE_string(name="model_dir",
                       default=None,
                       help="A string, representing the folder path of saved model files")
tf.flags.DEFINE_string(name="export_dir",
                       default=None,
                       help="A string, representing the basic folder path of export .pb files")
# ----------Hyperparameters of estimator (optional)----------
tf.flags.DEFINE_boolean(name="perform_valid_during_train",
                        default=True,
                        help="(optional) A boolean, instructing whether to perform validation on valid dataset during train phase")
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
                       help="A string, separating consecutive columns in a line of data file")
tf.flags.DEFINE_integer(name="field_size_numerical",
                        default=None,
                        help="An integer scalar, representing the number of numerical fields of dataset",
                        lower_bound=0,
                        upper_bound=None)
tf.flags.DEFINE_integer(name="field_size_categorical",
                        default=None,
                        help="An integer scalar, representing the number of categorical fields of dataset",
                        lower_bound=1,
                        upper_bound=None)
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
                        help="(optional) A boolean, instructing the dtype of both input function and model function. "
                             "If False, use tf.float32; if True, use tf.float64")
tf.flags.DEFINE_string(name="name_feat_vals_numerical",
                       default="inds",
                       help="(optional) A string, representing the name of numerical feature values in return dict")
tf.flags.DEFINE_string(name="name_feat_inds_categorical",
                       default="vals",
                       help="(optional) A string, representing the name of categorical feature indices in return dict")
# ----------Hyperparameters of model function----------
tf.flags.DEFINE_string(name="task",
                       default=None,
                       help="A string, representing the type of task (binary, multi or regression)")
tf.flags.DEFINE_integer(name="output_size",
                        default=None,
                        help="An integer scalar, representing the number of output units",
                        lower_bound=1,
                        upper_bound=None)
tf.flags.DEFINE_integer(name="feat_size_categorical",
                        default=None,
                        help="An integer scalar, representing the number of categorical features of dataset",
                        lower_bound=1,
                        upper_bound=None)
tf.flags.DEFINE_integer(name="embed_size",
                        default=None,
                        help="An integer scalar, representing the dimension of embedding vectors for all features",
                        lower_bound=1,
                        upper_bound=None)
tf.flags.DEFINE_list(name="deep_hidden_sizes",
                     default=None,
                     help="A list, containing the number of hidden units of each hidden layer in deep part (e.g. 128,64,32,16)")
tf.flags.DEFINE_list(name="dropouts",
                     default=None,
                     help="A list or None, containing the dropout rate of each hidden layer in dnn part (e.g. 0.4,0.3,0.2,0.1); "
                          "if None, don't use dropout operation for any hidden layer")
# ----------Hyperparameters of model function (optional)----------
tf.flags.DEFINE_boolean(name="use_global_bias",
                        default=True,
                        help="(optional) A boolean, instructing whether to use global bias in model inference")
tf.flags.DEFINE_boolean(name="use_deep_hidden_bias",
                        default=True,
                        help="(optional) A boolean, instructing whether to use bias of hidden layer units in deep part of model inference")
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
             field_size_numerical,
             field_size_categorical,
             batch_size,
             epochs,
             shuffle=True,
             buffer_size=100000,
             num_parallel_calls=4,
             dtype=tf.float32,
             name_feat_vals_numerical="vals_numerical",
             name_feat_inds_categorical="inds_categorical"):
    """
    The input function for loading sparse dataset in Deep & Cross format. The columns are separeted by argument
    delimiter(e.g. " ").
    Note:
        1. From left to right in each line, there contains three parts in order:
        label => numerical fields => categorical fields
            <label>
            <value 1 numerical> ... <value j_n numerical>
            <index 1 categorical> ... <index j_c categorical>
        2. Categorical fields group maintains an index-system independently.
        3. The feature type of each fields group:
            numerical fields(fixed length >= 0): dense fields
            categorical fields(fixed length > 1): each must be one-hot active field.
        4. The order of numerical fields must be fixed, and the order of categorical fields must be fixed too.
        5. The input function can be used for binary classification, multi classification and regression task:
            binary classification: <label> in {0, 1};
            multi classification: <label> in {0, K} (K is the number of total classes);
            regression: <label> in (-inf, inf).
    e.g.
        (field_size_numerical = 2, field_size_categorical = 6, feat_size_categorical = 300)
        0 0.172351 0.413592 1 14 20 134 231 293
        1 0.314512 0.871236 4 17 78 104 280 298
        For the first sample:
        0:                      label
        0.172351                value of 1-th numerical field
        0.413592                value of 2-th numerical field
        1 14 20 134 231 293     indices of categorical fields

    Parameters
    ----------
    :param filenames: list
        A list of string, containing one or more paths of filenames.
    :param delimiter: str
        A str, separating consecutive columns in data files.
    :param field_size_numerical: int
        An integer scalar, representing the number of numerical fields(namely the number of numerical features) of dataset.
    :param field_size_categorical: int
        An integer scalar, representing the number of categorical fields(number of categorical columns before one-hot encoding) of dataset.
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
    :param name_feat_vals_numerical: str, optional
        A string, representing the name of numerical feature values in return dict.
    :param name_feat_inds_categorical: str, optional
        A string, representing the name of categorical feature indices in return dict.

    Returns
    -------
    :return: dict
        A dict of two Tensors, representing features(including feature indices and feature values) in a single batch.
        {
            <name_feat_vals_numerical>: tf.Tensor of numerical feature values in shape of (None, field_size_numerical),
            <name_feat_inds_categorical>: tf.Tensor of categorical feature indices in shape of (None, field_size_categorical)
        }
    :return: Tensor
        A Tensor in shape of (None), representing labels in a single batch.
    """
    def map_func(line):
        columns = tf.string_split(source=[line], delimiter=delimiter, skip_empty=False)
        # ----------Process label----------
        label = tf.string_to_number(string_tensor=columns[0], out_type=dtype)
        # ----------Process numerical fields----------
        vals_numerical = tf.string_to_number(string_tensor=columns[1: 1 + field_size_numerical],
                                             out_type=dtype) # A tensor in shape of (field_size_numerical)
        # ----------Process categorical fields----------
        inds_categorical = tf.string_to_number(string_tensor=columns[-field_size_categorical: ],
                                               out_type=tf.int32) # A tensor in shape of (field_size_categorical)
        feats = {
            name_feat_vals_numerical: vals_numerical,
            name_feat_inds_categorical: inds_categorical
        }
        return feats, label

    dataset = tf.data.TextLineDataset(filenames=filenames, buffer_size=buffer_size). \
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
    """Model function of Wide & Deep Network(WDN) for predictive analytics of high dimensional sparse data.

    Args of dict params:
        task: str
            A string, representing the type of task.
            Note:
                it must take value from ["binary", "multi", "regression"];
                it instruct the type of loss function:
                    "binary": sigmoid cross-entropy;
                    "multi": softmax cross-entropy;
                    "regression": mean squared error.
        output_size: int
            An integer scalar, representing the number of output units.
            Note:
                it must be correspond to <task>:
                    task == "binary": output_size must be equal to 1;
                    task =="multi": output_size must be equal to the dimension of class distribution;
                    task == "regression": output_size must be equal to 1.
        field_size_numerical: int
            An integer scalar, representing the number of numerical fields(also is the number of numerical features) of dataset.
            Note:
                it must be consistent with <field_size_numerical> of input function.
        field_size_categorical: int
            An integer scalar, representing the number of categorical fields(number of categorical columns before one-hot encoding) of dataset.
            Note:
                it must be consistent with <field_size_categorical> of input function.
        feat_size_categorical: int
            An integer scalar, representing the number of categorical features(number of categorical columns after one-hot encoding) of dataset.
        embed_size: int
            An integer scalar, representing the dimension of embedding vectors for all categorical features.
        deep_hidden_sizes: list
            A list, containing the number of hidden units of each hidden layer in deep part.
            Note:
                it doesn't contain output layer of dnn part.
        dropouts: list or None
            If list, containing the dropout rate of each hidden layer in dnn part;
            If None, don't use dropout operation for any hidden layer.
            Note:
                if list, the length of <dropouts> must be equal to <hidden_sizes>.
        use_global_bias: bool
            A boolean, instructing whether to use global bias in output part of model inference.
        use_deep_hidden_bias: bool
            A boolean, instructing whether to use bias of hidden layer units in deep part of model inference.
        use_bn: bool
            A boolean, instructing whether to use batch normalization for each hidden layer in deep part.
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
        name_feat_vals_numerical: str, optional
            A string, representing the name of numerical feature values in return dict.
            Note:
                it must be consistent with <name_feat_vals_numerical> of input function.
        name_feat_inds_categorical: str, optional
            A string, representing the name of categorical feature indices in return dict.
            Note:
                it must be consistent with <name_feat_inds_categorical> of input function.
        reuse: bool
            A boolean, which takes value from [False, True, tf.AUTO_REUSE].
        seed: int or None
            If integer scalar, representing the random seed of tensorflow;
            If None, random choice.
    """
    # ----------Declare all hyperparameters from params----------
    task = params["task"]
    output_size = params["output_size"]
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
    dtype = params["dtype"]
    name_feat_vals_numerical = params["name_feat_vals_numerical"]
    name_feat_inds_categorical = params["name_feat_inds_categorical"]
    reuse = params["reuse"]
    seed = params["seed"]
    # -----Hyperparameters for threshold for binary classification task
    threshold = 0.5
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
    if dropouts != None:
        assert (len(dropouts) == len(deep_hidden_sizes))
    assert (dtype in [tf.float32, tf.float64])

    if seed != None:
        tf.set_random_seed(seed=seed)

    with tf.variable_scope(name_or_scope="inference", reuse=reuse):
        with tf.name_scope(name="inputs"):
            valsn = features[name_feat_vals_numerical]
            indsc = features[name_feat_inds_categorical]
            batch = tf.shape(input=valsn)[0]
            dim = field_size_numerical + field_size_categorical * embed_size

        with tf.name_scope(name="embed-and-concat"):
            V = tf.get_variable(name="V",
                                shape=[feat_size_categorical, embed_size],
                                dtype=dtype,
                                initializer=xavier_initializer(uniform=False, seed=seed, dtype=dtype),
                                regularizer=None)
            v = tf.nn.embedding_lookup(params=V, ids=indsc) # A tensor in shape of (None, field_size_categorical, embed_size)
            v = tf.reshape(tensor=v, shape=[-1, field_size_categorical * embed_size])
            ydeep = tf.reshape(tensor=tf.concat(values=[valsn, v], axis=-1), shape=[batch, dim])

        with tf.name_scope(name="deep-part"):
            for l in range(len(deep_hidden_sizes)):
                # -----The order for each hidden layer is: matmul => bn => relu => dropout => matmul => ...
                ydeep = tf.layers.dense(inputs=ydeep,
                                        units=deep_hidden_sizes[l],
                                        activation=None,
                                        use_bias=use_deep_hidden_bias,
                                        kernel_initializer=xavier_initializer(uniform=True, seed=seed, dtype=dtype),
                                        bias_initializer=tf.zeros_initializer(dtype=dtype),
                                        kernel_regularizer=l2_regularizer(scale=lamb),
                                        bias_regularizer=l2_regularizer(scale=lamb),
                                        trainable=True,
                                        name="deep-dense-hidden-{}".format(l))
                if use_bn == True:
                    ydeep = tf.layers.batch_normalization(inputs=ydeep,
                                                          axis=-1,
                                                          momentum=0.99,
                                                          epsilon=1e-3,
                                                          center=True,
                                                          scale=True,
                                                          beta_initializer=tf.zeros_initializer(dtype=dtype),
                                                          gamma_initializer=tf.ones_initializer(dtype=dtype),
                                                          moving_mean_initializer=tf.zeros_initializer(dtype=dtype),
                                                          moving_variance_initializer=tf.ones_initializer(dtype=dtype),
                                                          beta_regularizer=None,
                                                          gamma_regularizer=None,
                                                          training=(mode == tf.estimator.ModeKeys.TRAIN),
                                                          trainable=True,
                                                          name="deep-bn-hidden-{}".format(l))
                ydeep = tf.nn.relu(features=ydeep)
                if dropouts != None:
                    ydeep = tf.layers.dropout(inputs=ydeep,
                                              rate=dropouts[l],
                                              seed=seed,
                                              training=(mode == tf.estimator.ModeKeys.TRAIN))
            # -----output layer of deep part-----
            ydeep = tf.layers.dense(inputs=ydeep,
                                    units=output_size,
                                    activation=None,
                                    use_bias=False,
                                    kernel_initializer=xavier_initializer(uniform=True, seed=seed, dtype=dtype),
                                    kernel_regularizer=l2_regularizer(scale=lamb),
                                    trainable=True,
                                    name="deep-output") # A tensor in shape of (None, output_size)

        with tf.name_scope(name="wide-part"):
            ywide_n = tf.layers.dense(inputs=valsn,
                                      units=output_size,
                                      activation=None,
                                      use_bias=False,
                                      kernel_initializer=xavier_initializer(uniform=True, seed=seed, dtype=dtype),
                                      kernel_regularizer=l2_regularizer(scale=lamb),
                                      trainable=True,
                                      name="wide-output-numerical") # A tensor in shape of (None, output_size)
            WC = tf.get_variable(name="wide-output-categorical",
                                 shape=[feat_size_categorical, output_size],
                                 dtype=dtype,
                                 initializer=xavier_initializer(uniform=False, seed=seed, dtype=dtype),
                                 regularizer=l2_regularizer(scale=lamb))
            wc = tf.nn.embedding_lookup(params=WC, ids=indsc) # A tensor in shape of (None, field_size_categorical, output_size)
            ywide_c = tf.reduce_sum(input_tensor=wc, axis=1, keepdims=False) # A tensor in shape of (None, output_size)
            ywide = tf.add(x=ywide_n, y=ywide_c) # A tensor in shape of (None, output_size)

        with tf.name_scope(name="global-bias"):
            b = tf.get_variable(name="global-bias",
                                shape=[output_size],
                                dtype=dtype,
                                initializer=tf.zeros_initializer(dtype=dtype),
                                regularizer=None,
                                trainable=use_global_bias)

        with tf.name_scope(name="output"):
            if use_global_bias == True:
                logits = ywide + ydeep + b # A tensor in shape of (None, output_size)
            else:
                logits = ywide + ydeep # A tensor in shape of (None, output_size)

            if task == "binary":
                logits = tf.squeeze(input=logits, axis=1) # A tensor in shape of (None)
                probs = tf.nn.sigmoid(x=logits)
                classes = tf.cast(x=tf.greater(x=tf.nn.sigmoid(x=logits), y=threshold), dtype=tf.int32)
                predictions = {
                    name_probability_output: probs,
                    name_classification_output: classes
                }
            elif task == "multi":
                probs_dist = tf.nn.softmax(logits=logits, axis=-1)
                classes = tf.argmax(input=logits, axis=-1, output_type=tf.int32)
                predictions = {
                    name_probability_output: probs_dist,
                    name_classification_output: classes
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
                "confusion-matrix": tf.confusion_matrix(labels=labels,
                                                        predictions=predictions[name_classification_output],
                                                        num_classes=output_size)
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
    elif optimizer == "sgd-exp-decay":
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
    elif optimizer == "momentum-exp-decay":
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
    elif optimizer == "nesterov-exp-decay":
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
    # ----------Declare all hyperparameters of terminal interface----------
    phase = FLAGS.phase
    model_dir = FLAGS.model_dir
    export_dir = FLAGS.export_dir
    perform_valid_during_train = FLAGS.perform_valid_during_train # (optional) Note: if use validation dataset during train phase, the valid set and train set are in same data_dir
    log_step_count_steps = FLAGS.log_step_count_steps # (optional)
    save_checkpoints_steps = FLAGS.save_checkpoints_steps # (optional)
    keep_checkpoint_max = FLAGS.keep_checkpoint_max # (optional)
    data_dir = FLAGS.data_dir
    delimiter = FLAGS.delimiter
    field_size_numerical = FLAGS.field_size_numerical
    field_size_categorical = FLAGS.field_size_categorical
    batch_size = FLAGS.batch_size
    epochs = FLAGS.epochs
    shuffle = FLAGS.shuffle # (optional)
    buffer_size = FLAGS.buffer_size # (optional)
    num_parallel_calls = FLAGS.num_parallel_calls # (optional)
    use_dtype_high_precision = FLAGS.use_dtype_high_precision # (optional)
    name_feat_vals_numerical = FLAGS.name_feat_vals_numerical # (optional)
    name_feat_inds_categorical = FLAGS.name_feat_inds_categorical # (optional)
    task = FLAGS.task
    output_size = FLAGS.output_size
    feat_size_categorical = FLAGS.feat_size_categorical
    embed_size = FLAGS.embed_size
    deep_hidden_sizes = FLAGS.deep_hidden_sizes
    dropouts = FLAGS.dropouts
    use_global_bias = FLAGS.use_global_bias # (optional)
    use_deep_hidden_bias = FLAGS.use_deep_hidden_bias # (optional)
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

    hparams = {
        "task": task,
        "output_size": output_size,
        "field_size_numerical":  field_size_numerical,
        "field_size_categorical": field_size_categorical,
        "feat_size_categorical": feat_size_categorical,
        "embed_size": embed_size,
        "deep_hidden_sizes": deep_hidden_sizes,
        "dropouts": dropouts,
        "use_global_bias": use_global_bias,
        "use_deep_hidden_bias": use_deep_hidden_bias,
        "use_bn": use_bn,
        "lamb": lamb,
        "optimizer": optimizer,
        "learning_rate": learning_rate,
        "dtype": dtype,
        "name_feat_vals_numerical": name_feat_vals_numerical,
        "name_feat_inds_categorical": name_feat_inds_categorical,
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
        if perform_valid_during_train == True:
            filenames_valid = tf.gfile.Glob(filename=os.path.join(data_dir, PREFIX_EVAL_FILE + "*"))
            train_spec = tf.estimator.TrainSpec(input_fn=lambda : input_fn(filenames=filenames_train,
                                                                           delimiter=delimiter,
                                                                           field_size_numerical=field_size_numerical,
                                                                           field_size_categorical=field_size_categorical,
                                                                           batch_size=batch_size,
                                                                           epochs=epochs,
                                                                           shuffle=shuffle,
                                                                           buffer_size=buffer_size,
                                                                           num_parallel_calls=num_parallel_calls,
                                                                           dtype=dtype,
                                                                           name_feat_vals_numerical=name_feat_vals_numerical,
                                                                           name_feat_inds_categorical=name_feat_inds_categorical),
                                                max_steps=None)
            eval_spec = tf.estimator.EvalSpec(input_fn=lambda : input_fn(filenames=filenames_valid,
                                                                         delimiter=delimiter,
                                                                         field_size_numerical=field_size_numerical,
                                                                         field_size_categorical=field_size_categorical,
                                                                         batch_size=batch_size,
                                                                         epochs=1,
                                                                         shuffle=False,
                                                                         buffer_size=buffer_size,
                                                                         num_parallel_calls=num_parallel_calls,
                                                                         dtype=dtype,
                                                                         name_feat_vals_numerical=name_feat_vals_numerical,
                                                                         name_feat_inds_categorical=name_feat_inds_categorical),
                                              steps=None,
                                              start_delay_secs=120,
                                              throttle_secs=600)
            tf.estimator.train_and_evaluate(estimator=estimator, train_spec=train_spec, eval_spec=eval_spec)
        else:
            estimator.train(input_fn=lambda : input_fn(filenames=filenames_train,
                                                       delimiter=delimiter,
                                                       field_size_numerical=field_size_numerical,
                                                       field_size_categorical=field_size_categorical,
                                                       batch_size=batch_size,
                                                       epochs=epochs,
                                                       shuffle=shuffle,
                                                       buffer_size=buffer_size,
                                                       num_parallel_calls=num_parallel_calls,
                                                       dtype=dtype,
                                                       name_feat_vals_numerical=name_feat_vals_numerical,
                                                       name_feat_inds_categorical=name_feat_inds_categorical),
                            steps=None,
                            max_steps=None)
    elif phase == "eval":
        filenames_eval = tf.gfile.Glob(filename=os.path.join(data_dir, PREFIX_EVAL_FILE + "*"))
        estimator.evaluate(input_fn=lambda : input_fn(filenames=filenames_eval,
                                                      delimiter=delimiter,
                                                      field_size_numerical=field_size_numerical,
                                                      field_size_categorical=field_size_categorical,
                                                      batch_size=batch_size,
                                                      epochs=1,
                                                      shuffle=False,
                                                      buffer_size=buffer_size,
                                                      num_parallel_calls=num_parallel_calls,
                                                      dtype=dtype,
                                                      name_feat_vals_numerical=name_feat_vals_numerical,
                                                      name_feat_inds_categorical=name_feat_inds_categorical))
    elif phase == "predict":
        filenames_predict = tf.gfile.Glob(filename=os.path.join(data_dir, PREFIX_PREDICT_FILE + "*"))
        p = estimator.predict(input_fn=lambda : input_fn(filenames=filenames_predict,
                                                         delimiter=delimiter,
                                                         field_size_numerical=field_size_numerical,
                                                         field_size_categorical=field_size_categorical,
                                                         batch_size=batch_size,
                                                         epochs=1,
                                                         shuffle=False,
                                                         buffer_size=buffer_size,
                                                         num_parallel_calls=num_parallel_calls,
                                                         dtype=dtype,
                                                         name_feat_vals_numerical=name_feat_vals_numerical,
                                                         name_feat_inds_categorical=name_feat_inds_categorical))
        # -----Usage demo, still need to be accomplished-----
        for ele in p:
            print(ele)
    elif phase == "export":
        features = {
            name_feat_vals_numerical: tf.placeholder(dtype=dtype,
                                                     shape=[None, field_size_numerical],
                                                     name=name_feat_vals_numerical),
            name_feat_inds_categorical: tf.placeholder(dtype=tf.int32,
                                                       shape=[None, field_size_categorical],
                                                       name=name_feat_inds_categorical)
        }
        serving_input_receiver_fn = tf.estimator.export.build_raw_serving_input_receiver_fn(features=features)
        estimator.export_savedmodel(export_dir_base=export_dir, serving_input_receiver_fn=serving_input_receiver_fn)
    else:
        raise NotImplementedError("Argument <phase> value: {} is not supported.".format(phase))

if __name__ == '__main__':
    tf.logging.set_verbosity(v=tf.logging.INFO)
    tf.app.run(main=main)