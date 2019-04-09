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

    if seed != None:
        tf.set_random_seed(seed=seed)

    with tf.variable_scope(name_or_scope="inference", reuse=reuse):
        with tf.name_scope(name="inputs"):
            valsn = features[name_feat_vals_numerical]
            indsc = features[name_feat_inds_categorical]
