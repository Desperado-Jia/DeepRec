"""
Created on 2019/03/25 02:52
author: Tong Jia
email: cecilio.jia@gmail.com
description:
    An implementation of Deep Interest Network for Click-Through Rate Prediction (DIN).
    See algorithm and hyperparameter details:
        [Zhou et al., 2018](https://arxiv.org/pdf/1706.06978.pdf)
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
tf.flags.DEFINE_integer(name="field_size_user_profile",
                        default=None,
                        help="An integer scalar, representing the number of fields belong to user profile",
                        lower_bound=1,
                        upper_bound=None)
tf.flags.DEFINE_integer(name="field_size_item_profile",
                        default=None,
                        help="An integer scalar, representing the number of fields belong to item profile",
                        lower_bound=1,
                        upper_bound=None)
tf.flags.DEFINE_integer(name="field_size_context",
                        default=None,
                        help="An integer scalar, representing the number of fields belong to context",
                        lower_bound=1,
                        upper_bound=None)
tf.flags.DEFINE_integer(name="behaviors_size",
                        default=None,
                        help="An integer scalar, representing the length of user behaviors list",
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
tf.flags.DEFINE_string(name="name_feat_inds_user",
                       default="user_profile_inds",
                       help="(optional) A string, representing the name of feature indices of user profile fields in return dict")
tf.flags.DEFINE_string(name="name_feat_vals_user",
                       default="user_profile_vals",
                       help="(optional) A string, representing the name of feature values of user profile fields in return dict")
tf.flags.DEFINE_string(name="name_feat_inds_item",
                       default="item_profile_inds",
                       help="(optional) A string, representing the name of feature indices of item profile fields in return dict")
tf.flags.DEFINE_string(name="name_feat_vals_item",
                       default="item_profile_vals",
                       help="(optional) A string, representing the name of feature values of item profile fields in return dict")
tf.flags.DEFINE_string(name="name_feat_inds_context",
                       default="cont_inds",
                       help="(optional) A string, representing the name of feature indices of context fields in return dict")
tf.flags.DEFINE_string(name="name_feat_vals_context",
                       default="cont_vals",
                       help="(optional) A string, representing the name of feature values of context fields in return dict")
tf.flags.DEFINE_string(name="name_feat_inds_candidate",
                       default="cand_inds",
                       help="(optional) A string, representing the name of feature index of candidate in return dict")
tf.flags.DEFINE_string(name="name_feat_inds_behaviors",
                       default="beha_inds",
                       help="(optional) A string, representing the name of feature indices of user behaviors in return dict")
# ----------Hyperparameters of model function----------
tf.flags.DEFINE_string(name="task",
                       default=None,
                       help="A string, representing the type of task (binary or regression)")
tf.flags.DEFINE_integer(name="output_size",
                        default=None,
                        help="An integer scalar, representing the number of output units",
                        lower_bound=1,
                        upper_bound=None)
tf.flags.DEFINE_integer(name="feat_size_user_profile",
                        default=None,
                        help="An integer scalar, representing the number of features belong to user profile",
                        lower_bound=1,
                        upper_bound=None)
tf.flags.DEFINE_integer(name="feat_size_item_profile",
                        default=None,
                        help="An integer scalar, representing the number of features belong to item profile",
                        lower_bound=1,
                        upper_bound=None)
tf.flags.DEFINE_integer(name="feat_size_context",
                        default=None,
                        help="An integer scalar, representing the number of features belong to context",
                        lower_bound=1,
                        upper_bound=None)
tf.flags.DEFINE_integer(name="feat_size_id",
                        default=None,
                        help="An integer scalar, representing the number of ids of item(for candidate and user behaviors)",
                        lower_bound=1,
                        upper_bound=None)
tf.flags.DEFINE_integer(name="embed_size_user_profile",
                        default=None,
                        help="An integer scalar, representing the dimension of embedding vectors for all features belong to user profile fields",
                        lower_bound=1,
                        upper_bound=None)
tf.flags.DEFINE_integer(name="embed_size_item_profile",
                        default=None,
                        help="An integer scalar, representing the dimension of embedding vectors for all features belong to item profile fields",
                        lower_bound=1,
                        upper_bound=None)
tf.flags.DEFINE_integer(name="embed_size_context",
                        default=None,
                        help="An integer scalar, representing the dimension of embedding vectors for all features belong to context fields",
                        lower_bound=1,
                        upper_bound=None)
tf.flags.DEFINE_integer(name="embed_size_id",
                        default=None,
                        help="An integer scalar, representing the dimension of embedding vectors for candidate and all behaviors (item IDs)",
                        lower_bound=1,
                        upper_bound=None)
tf.flags.DEFINE_list(name="hidden_sizes",
                     default=None,
                     help="A list, containing the number of hidden units of each hidden layer in dnn part")
tf.flags.DEFINE_list(name="dropouts",
                     default=None,
                     help="A list, containing the dropout rate of each hidden layer in dnn part (e.g. 128,64,64,32); if None, don't use dropout operation for any hidden layer")
# ----------Hyperparameters of model function (optional)----------
tf.flags.DEFINE_boolean(name="use_softmax_norm_for_attention",
                        default=False,
                        help="A boolean, instructing whether to perform normalization with softmax on the output of attention scores between candidate embedding and user behaviors embeddings")
tf.flags.DEFINE_boolean(name="use_bn",
                        default=True,
                        help="(optional) A boolean, instructing whether to use batch normalization for each hidden layer in dnn part")
tf.flags.DEFINE_boolean(name="use_global_bias",
                        default=True,
                        help="(optional) A boolean, instructing whether to use global bias in model inference")
tf.flags.DEFINE_boolean(name="use_hidden_bias",
                        default=True,
                        help="(optional) A boolean, instructing whether to use bias of hidden layer units in model inference")
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
             field_size_user_profile,
             field_size_item_profile,
             field_size_context,
             behaviors_size,
             shuffle=True,
             buffer_size=100000,
             num_parallel_calls=4,
             dtype=tf.float32,
             name_feat_inds_user="user_profile_inds",
             name_feat_vals_user="user_profile_vals",
             name_feat_inds_item="item_profile_inds",
             name_feat_vals_item="item_profile_vals",
             name_feat_inds_context="cont_inds",
             name_feat_vals_context="cont_vals",
             name_feat_inds_candidate = "cand_inds",
             name_feat_inds_behaviors = "beha_inds"):
    """
    The input function for loading sparse dataset in Deep Interest Network format. The columns are separeted by argument
    delimiter(e.g. " ").
    Note:
        1. From left to right in each line, there contains six parts in order:
        label => user fields => item fields => context fields => candidate index => user behaviors indices
            <label>
            <index 1 user>:<value 1 user> ... <index j_u user>:<value j_u user>
            <index 1 item>:<value 1 item> ... <index j_i item>:<value j_i item>
            <index 1 context>:<value 1 context> ... <index j_c context>:<value j_c context>
            <index candidate>
            <index 1 behaviors> ... <index j_bmax behaviors>
        2. Each fields group maintains a index system independently, namely:
            a. There exist index 1 in both user fields group and item fields group
            b. The meaning of the first-index value in user fields group is different from the first-index value in item fields group
        3. The feature type of each fields group:
            user fields(fixed length > 1): dense field, one-hot active field
            item fields(fixed length > 1): dense field, one-hot active field
            context fields(fixed length > 1): dense field, one-hot active field
            candidate field(fixed length = 1): one-hot active field
            user behaviors indices(fixed length = behaviors_size): multi-hot active field
        4. The order of behaviors indices is unnecessary because of sum pooling operation later.
        5. The input function can be used for binary classification, multi classification and regression task:
            binary classification: <label> in {0, 1};
            multi classification: <label> in {0, K} (K is the number of total classes);
            regression: <label> in (-inf, inf).
    e.g.
        (field_size_user_profile = 2, field_size_item_profile = 3, field_size_context = 1, behaviors_size = 5)
        1 0:1 6:1 0:1 10:1 17:0.456127 0:1 14 1 4 5 8 9
        0 0:1 9:1 3:1 7:1 17:0.077712 3:1 2 7 3 6 18 21 88 83 13 22 71
        if the length of user behaviors is greater than <behaviors_size>, we will select first <behaviors_size> element from left to right.
        For the first sample:
        1:                      label
        0:1 6:1:                features of user profile
        0:1 10:1 17:0.456127:   features of item profile
        0:1:                    features of context
        14:                     index of candidate
        1 4 5 8 9:              indices of user behaviors

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
    :param field_size_user_profile: int
        An integer scalar, representing the number of fields(number of columns before one-hot encoding) of user profile.
    :param field_size_item_profile: int
        An integer scalar, representing the number of fields(number of columns before one-hot encoding) of item profile.
    :param field_size_context: int
        An integer scalar, representing the number of fields(number of columns before one-hot encoding) of context.
    :param behaviors_size: int
        An integer scalar, representing the length of user behaviors list.
    :param shuffle: boolean, optional
        A boolean(defaults to True), instructing whether to randomly shuffle the elements of this dataset.
    :param buffer_size: int, optional
        An integer scalar(defaults to 100000), denoting the number of bytes to buffer.
    :param num_parallel_calls: int, optional
        An integer scalar(defaults to 4), representing the number elements to process in parallel.
    :param dtype: tf.Dtype, optional
        A tf.DType(defaults to tf.float32), representing the numeric type of values. it always takes value from [tf.float32, tf.float64].
    :param name_feat_inds_user: str, optional
        A string, representing the name of feature indices of user profile fields in return dict.
    :param name_feat_vals_user: str, optional
        A string, representing the name of feature values of user profile fields in return dict.
    :param name_feat_inds_item: str, optional
        A string, representing the name of feature indices of item profile fields in return dict.
    :param name_feat_vals_item: str, optional
        A string, representing the name of feature values of item profile fields in return dict.
    :param name_feat_inds_context: str, optional
        A string, representing the name of feature indices of context fields in return dict.
    :param name_feat_vals_context: str, optional
        A string, representing the name of feature values of context fields in return dict.
    :param name_feat_inds_candidate: str, optional
        A string, representing the name of feature index of candidate in return dict.
    :param name_feat_inds_behaviors: str, optional
        A string, representing the name of feature indices of user behaviors in return dict.

    Returns
    -------
    :return: dict
        A dict of eight Tensors, representing features(including feature indices and feature values) in a single batch.
        {
            <name_feat_inds_user>: tf.Tensor of feature indices of user profile fields in shape of (None, field_size_user_profile),
            <name_feat_vals_user>: tf.Tensor of feature values of user profile fields in shape of (None, field_size_user_profile),
            <name_feat_inds_item>: tf.Tensor of feature indices of item profile fields in shape of (None, field_size_item_profile),
            <name_feat_vals_item>: tf.Tensor of feature values of item profile fields in shape of (None, field_size_item_profile),
            <name_feat_inds_context>: tf.Tensor of feature indices of context fields in shape of (None, field_size_context),
            <name_feat_vals_context>: tf.Tensor of feature values of context in shape of (None, field_size_context)
            <name_feat_inds_candidate>: tf.Tensor of feature index of candidate in shape of (None)
            <name_feat_inds_behaviors>: tf.Tensor of feature indices of user behaviors in shape of (None, behaviors_size)
        }
    :return Tensor
        A Tensor in shape of (None), representing labels in a single batch.
    """
    START_COLUMN_FEATS_USER = 1
    END_COLUMN_FEATS_USER = START_COLUMN_FEATS_USER + field_size_user_profile
    START_COLUMN_FEATS_ITEM = END_COLUMN_FEATS_USER
    END_COLUMN_FEATS_ITEM = START_COLUMN_FEATS_ITEM + field_size_item_profile
    START_COLUMN_FEATS_CONT = END_COLUMN_FEATS_ITEM
    END_COLUMN_FEATS_CONT = START_COLUMN_FEATS_CONT + field_size_context
    START_COLUMN_FEAT_CANDIDATE = END_COLUMN_FEATS_CONT
    START_COLUMN_FEAT_USER_BEHAVIORS = START_COLUMN_FEAT_CANDIDATE + 1
    END_COLUMN_FEAT_USER_BEHAVIORS = START_COLUMN_FEAT_USER_BEHAVIORS + behaviors_size
    def map_func(line):
        columns = tf.string_split(source=[line], delimiter=delimiter, skip_empty=False).values
        # ----------Process label----------
        label = tf.string_to_number(string_tensor=columns[0], out_type=dtype)
        # ----------Process user profile fields----------
        splits_user = tf.string_split(source=columns[START_COLUMN_FEATS_USER: END_COLUMN_FEATS_USER],
                                      delimiter=separator,
                                      skip_empty=False)
        feats_user = tf.reshape(tensor=splits_user.values, shape=splits_user.dense_shape)
        inds_user, vals_user = tf.split(value=feats_user, num_or_size_splits=2, axis=1) # Two tensors in shape of (field_size_user_profile, 1)
        feat_inds_user = tf.squeeze(input=tf.string_to_number(string_tensor=inds_user, out_type=tf.int32), axis=1) # A tensor in shape of (field_size_user_profile)
        feat_vals_user = tf.squeeze(input=tf.string_to_number(string_tensor=vals_user, out_type=dtype), axis=1) # A tensor in shape of (field_size_user_profile)
        # ----------Process item profile fields (not contain item index)----------
        splits_item = tf.string_split(source=columns[START_COLUMN_FEATS_ITEM: END_COLUMN_FEATS_ITEM],
                                      delimiter=separator,
                                      skip_empty=False)
        feats_item = tf.reshape(tensor=splits_item.values, shape=splits_item.dense_shape)
        inds_item, vals_item = tf.split(value=feats_item, num_or_size_splits=2, axis=1)
        feat_inds_item = tf.squeeze(input=tf.string_to_number(string_tensor=inds_item, out_type=tf.int32), axis=1) # A tensor in shape of (field_size_item_profile)
        feat_vals_item = tf.squeeze(input=tf.string_to_number(string_tensor=vals_item, out_type=dtype), axis=1) # A tensor in shape of (field_size_item_profile)
        # ----------Process context fields----------
        splits_cont = tf.string_split(source=columns[START_COLUMN_FEATS_CONT: END_COLUMN_FEATS_CONT],
                                      delimiter=separator,
                                      skip_empty=False)
        feats_cont = tf.reshape(tensor=splits_cont.values, shape=splits_cont.dense_shape)
        inds_cont, vals_cont = tf.split(value=feats_cont, num_or_size_splits=2, axis=1)
        feat_inds_cont = tf.squeeze(input=tf.string_to_number(string_tensor=inds_cont, out_type=tf.int32), axis=1) # A tensor in shape of (field_size_context)
        feat_vals_cont = tf.squeeze(input=tf.string_to_number(string_tensor=vals_cont, out_type=dtype), axis=1) # A tensor in shape of (field_size_context)
        # ----------Process candidate item index----------
        feat_inds_cand = tf.string_to_number(string_tensor=columns[START_COLUMN_FEAT_CANDIDATE], out_type=tf.int32)
        # ----------Process user behaviors (associated item indices)----------
        feat_inds_beha = tf.string_to_number(string_tensor=columns[START_COLUMN_FEAT_USER_BEHAVIORS: END_COLUMN_FEAT_USER_BEHAVIORS], out_type=tf.int32)

        feats = {
            name_feat_inds_user: feat_inds_user,
            name_feat_vals_user: feat_vals_user,
            name_feat_inds_item: feat_inds_item,
            name_feat_vals_item: feat_vals_item,
            name_feat_inds_context: feat_inds_cont,
            name_feat_vals_context: feat_vals_cont,
            name_feat_inds_candidate: feat_inds_cand,
            name_feat_inds_behaviors: feat_inds_beha
        }
        return feats, label

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
    """Model function of Deep interest network(DIN) for predictive analytics of high dimensional sparse data.

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
        field_size_user_profile: int
            An integer scalar, representing the number of fields(number of columns before one-hot encoding) of user profile.
            Note:
                it must be consistent with <field_size_user_profile> of input function.
        field_size_item_profile: int
            An integer scalar, representing the number of fields(number of columns before one-hot encoding) of item profile.
            Note:
                it must be consistent with <field_size_item_profile> of input function.
        field_size_context: int
            An integer scalar, representing the number of fields(number of columns before one-hot encoding) of context.
            Note:
                it must be consistent with <field_size_context> of input function.
        feat_size_user_profile: int
            An integer scalar, representing the number of features(number of columns after one-hot encoding) of user profile.
        feat_size_item_profile: int
            An integer scalar, representing the number of features(number of columns after one-hot encoding) of item profile.
        feat_size_context: int
            An integer scalar, representing the number of features(number of columns after one-hot encoding) of context.
        feat_size_id: int
            An integer scalar, representing the number of features(number of id after one-hot encoding) of item(for candidate and user behaviors).
        embed_size_user_profile: int
            An integer scalar, representing the dimension of embedding vectors for all features belong to user profile fields.
        embed_size_item_profile: int
            An integer scalar, representing the dimension of embedding vectors for all features belong to item profile fields.
        embed_size_context: int
            An integer scalar, representing the dimension of embedding vectors for all features belong to context fields.
        embed_size_id: int
            An integer scalar, representing the dimension of embedding vectors for candidate and all behaviors (item IDs).
        hidden_sizes: list
            A list, containing the number of hidden units of each hidden layer in dnn part.
            Note:
                it doesn't contain output layer of dnn part.
        dropouts: list or None
            If list, containing the dropout rate of each hidden layer in dnn part;
            If None, don't use dropout operation for any hidden layer.
            Note:
                if list, the length of <dropouts> must be equal to <hidden_sizes>.
        use_softmax_norm_for_attention: bool
            A boolean, instructing whether to perform normalization with softmax on the output of attention scores between
            candidate embedding and user behaviors embeddings.
            Note:
                In raw paper, it don't suggest to perform normalization with softmax on the output of attention scores.
                For example, if one user’s historical behaviors contain 90% clothes and 10% electronics. Given two
                candidate ads of T-shirt and phone, T-shirt activates most of the historical behaviors belonging to
                clothes and may get larger value of v_U (higher intensity of interest) than phone. Traditional attention
                methods lose the resolution on the numerical scale of v_U by normalizing of the output of a(·).
                So:
                    if all behaviors and all candidates are almost similar(come from the same category), we can set
                    use_softmax_norm_for_attention == True;
                    elso, we can use_softmax_norm_for_attention == False to get better representation ability of
                    intensity of interest.
        use_bn: bool
            A boolean, instructing whether to use batch normalization for each hidden layer in dnn part.
        use_hidden_bias: bool
            A boolean, instructing whether to use bias of hidden layer units in model inference.
        use_global_bias: bool
            A boolean, instructing whether to use global bias in model inference.
        lamb: float
            A float scalar, representing the coefficient of regularization term (the larger the value of lamb, the stronger the penalty is).
            Note:
                Here regularization is only used for global bias.
        optimizer: str
            A string, representing the type of optimizer.
        learning_rate: float
            A float scalar, representing the learning rate of optimizer.
        dtype: tf.Dtype
            A tf.DType, representing the numeric type of values.
            Note:
                it must take value from [tf.float32, tf.float64];
                it must be consistent with <dtype> of input function.
        name_feat_inds_user: str, optional
            A string, representing the name of feature indices of user profile fields in return dict.
            Note:
                it must be consistent with <name_feat_inds_user> of input function.
        name_feat_vals_user: str, optional
            A string, representing the name of feature values of user profile fields in return dict.
            Note:
                it must be consistent with <name_feat_vals_user> of input function.
        name_feat_inds_item: str, optional
            A string, representing the name of feature indices of item profile fields in return dict.
            Note:
                it must be consistent with <name_feat_inds_item> of input function.
        name_feat_vals_item: str, optional
            A string, representing the name of feature values of item profile fields in return dict.
            Note:
                it must be consistent with <name_feat_vals_item> of input function.
        name_feat_inds_context: str, optional
            A string, representing the name of feature indices of context fields in return dict.
            Note:
                it must be consistent with <name_feat_inds_context> of input function.
        name_feat_vals_context: str, optional
            A string, representing the name of feature values of context fields in return dict.
            Note:
                it must be consistent with <name_feat_vals_context> of input function.
        name_feat_inds_candidate: str, optional
            A string, representing the name of feature index of candidate in return dict.
            Note:
                it must be consistent with <name_feat_inds_candidate> of input function.
        name_feat_inds_behaviors: str, optional
            A string, representing the name of feature indices of user behaviors in return dict.
            Note:
                it must be consistent with <name_feat_inds_behaviors> of input function.
        reuse: bool
            A boolean, which takes value from [False, True, tf.AUTO_REUSE].
        seed: int or None
            If integer scalar, representing the random seed of tensorflow;
            If None, random choice.
    """
    # ----------Declare all hyperparameters from params----------
    task = params["task"]
    output_size = params["output_size"]
    field_size_user_profile = params["field_size_user_profile"]
    field_size_item_profile = params["field_size_item_profile"]
    field_size_context = params["field_size_context"]
    feat_size_user_profile = params["feat_size_user_profile"]
    feat_size_item_profile = params["feat_size_item_profile"]
    feat_size_context = params["feat_size_context"]
    feat_size_id = params["feat_size_id"]
    embed_size_user_profile = params["embed_size_user_profile"]
    embed_size_item_profile = params["embed_size_item_profile"]
    embed_size_context = params["embed_size_context"]
    embed_size_id = params["embed_size_id"]
    hidden_sizes = params["hidden_sizes"]
    dropouts = params["dropouts"]
    use_softmax_norm_for_attention = params["use_softmax_norm_for_attention"]
    use_bn = params["use_bn"]
    use_hidden_bias = params["use_hidden_bias"]
    use_global_bias = params["use_global_bias"]
    lamb = params["lamb"]
    optimizer = params["optimizer"]
    learning_rate = params["learning_rate"]
    dtype = params["dtype"]
    name_feat_inds_user = params["name_feat_inds_user"]
    name_feat_vals_user = params["name_feat_vals_user"]
    name_feat_inds_item = params["name_feat_inds_item"]
    name_feat_vals_item = params["name_feat_vals_item"]
    name_feat_inds_context = params["name_feat_inds_context"]
    name_feat_vals_context = params["name_feat_vals_context"]
    name_feat_inds_candidate = params["name_feat_inds_candidate"]
    name_feat_inds_behaviors = params["name_feat_inds_behaviors"]
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

    # ----------Build model inference----------
    with tf.variable_scope(name_or_scope="inference", reuse=reuse):
        with tf.name_scope(name="inputs"):
            ids_u = features[name_feat_inds_user] # A tensor in shape of (None, field_size_user_profile)
            x_u = features[name_feat_vals_user] # A tensor in shape of (None, field_size_user_profile)
            ids_i = features[name_feat_inds_item] # A tensor in shape of (None, field_size_item_profile)
            x_i = features[name_feat_vals_item] # A tensor in shape of (None, field_size_item_profile)
            ids_co = features[name_feat_inds_context]
            x_co = features[name_feat_vals_context]
            ids_ca = features[name_feat_inds_candidate]
            ids_be = features[name_feat_inds_behaviors]

        with tf.name_scope(name="embed-user-profile-layer"):
            Vu = tf.get_variable(name="Vu",
                                 shape=[feat_size_user_profile, embed_size_user_profile],
                                 dtype=dtype,
                                 initializer=tfc.layers.xavier_initializer(uniform=False, dtype=dtype),
                                 regularizer=None)
            vu = tf.nn.embedding_lookup(params=Vu, ids=ids_u) # A tensor in shape of (None, field_size_user_profile, embed_size_user_profile)

        with tf.name_scope(name="embed-item-profile-layer"):
            Vi = tf.get_variable(name="Vi",
                                 shape=[feat_size_item_profile, embed_size_item_profile],
                                 dtype=dtype,
                                 initializer=tfc.layers.xavier_initializer(uniform=False, dtype=dtype),
                                 regularizer=None)
            vi = tf.nn.embedding_lookup(params=Vi, ids=ids_i) # A tensor in shape of (None, field_size_item_profile, embed_size_item_profile)

        with tf.name_scope(name="embed-context-layer"):
            Vc = tf.get_variable(name="Vc",
                                 shape=[feat_size_context, embed_size_context],
                                 dtype=dtype,
                                 initializer=tfc.layers.xavier_initializer(uniform=False, dtype=dtype),
                                 regularizer=None)
            vc = tf.nn.embedding_lookup(params=Vc, ids=ids_co) # A tensor in shape of (None, field_size_context, embed_size_context)

        with tf.name_scope(name="embed-id-layer"):
            Vid = tf.get_variable(name="Vid",
                                  shape=[feat_size_id, embed_size_id],
                                  dtype=dtype,
                                  initializer=tfc.layers.xavier_initializer(uniform=False, dtype=dtype),
                                  regularizer=None)
            # -----embedding look up operation for candidate-----
            vca = tf.nn.embedding_lookup(params=Vid, ids=ids_ca) # A tensor in shape of (None, embed_size_id)
            # -----embedding look up operation for user behaviors-----
            vbe = tf.nn.embedding_lookup(params=Vid, ids=ids_be) # A tensor in shape of (None, behaviors_size, embed_size_id)

        with tf.name_scope(name="attention-id-layer"):
            # -----Key part of DIN model-----
            vca_reshape = tf.expand_dims(input=vca, axis=1) # A tensor in shape of (None, 1, embed_size_id)
            # We can use many methods to build up attention function such as dot-products, feed-forward network and so on, here we use dot-product method
            attention = tf.reduce_sum(input_tensor=tf.multiply(x=vca_reshape, y=vbe), # A tensor in shape of (None, behaviors_size, embed_size_id)
                                      axis=-1,
                                      keepdims=False) # A tensor in shape of (None, behaviors_size)
            if use_softmax_norm_for_attention == True:
                attention = tf.nn.softmax(logits=attention, axis=-1)
            attention = tf.expand_dims(input=attention, axis=-1) # A tensor in shape of (None, behaviors_size, 1)
            vembed = tf.reduce_sum(input_tensor=tf.multiply(x=attention, y=vbe), # A tensor in shape of (None, behaviors_size, embed_size_id)
                                   axis=1,
                                   keepdims=False) # A tensor in shape of (None, embed_size_id)

        with tf.name_scope(name="concat-flatten-layer"):
            vu = tf.reshape(tensor=vu, shape=[-1, field_size_user_profile * embed_size_user_profile])
            vi = tf.reshape(tensor=vi, shape=[-1, field_size_item_profile * embed_size_item_profile])
            vc = tf.reshape(tensor=vc, shape=[-1, field_size_context * embed_size_context])
            logits = tf.concat(values=[vu, vi, vc, vca, vembed], axis=1) # A tensor in shape of (None, dnn_input_size)

        with tf.name_scope(name="dnn-hidden-layer"):
            for l in range(len(hidden_sizes)):
                # -----The order for each hidden layer is: matmul => bn => relu => dropout => matmul => ...
                logits = tf.layers.dense(inputs=logits,
                                       units=hidden_sizes[l],
                                       activation=None,
                                       use_bias=use_hidden_bias,
                                       kernel_initializer=tfc.layers.xavier_initializer(uniform=True, dtype=dtype),
                                       bias_initializer=tf.zeros_initializer(dtype=dtype),
                                       kernel_regularizer=None, # *manual optional*
                                       bias_regularizer=None, # *manual optional*
                                       name="mlp-dense-hidden-{}".format(l))
                if use_bn == True:
                    logits = tf.layers.batch_normalization(inputs=logits,
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
                                                         name="mlp-bn-hidden-{}".format(l))

                logits = tf.nn.relu(features=logits)
                if dropouts != None:
                    logits = tf.layers.dropout(inputs=logits,
                                               rate=dropouts[l],
                                               seed=seed,
                                               training=(mode == tf.estimator.ModeKeys.TRAIN))

        with tf.name_scope(name="output"):
            logits = tf.layers.dense(inputs=logits,
                                     units=output_size,
                                     activation=None,
                                     use_bias=use_global_bias,
                                     kernel_initializer=tfc.layers.xavier_initializer(uniform=True, dtype=dtype),
                                     bias_initializer=tf.zeros_initializer(dtype=dtype),
                                     kernel_regularizer=None, # *manual optional*
                                     bias_regularizer=tfc.layers.l2_regularizer(scale=lamb), # *manual optional*
                                     name="mlp-dense-output") # A tensor in shape of (None, output_size)
            if task == "binary":
                logits = tf.squeeze(input=logits, axis=1) # A tensor in shape of (None)
                predictions = {
                    name_probability_output: tf.nn.sigmoid(x=logits),
                    name_classification_output: tf.cast(x=tf.round(x=tf.nn.sigmoid(x=logits)), dtype=tf.uint8)
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
    # ----------Declare all hyperparameters----------
    phase = FLAGS.phase
    model_dir = FLAGS.model_dir
    export_dir = FLAGS.export_dir
    log_step_count_steps = FLAGS.log_step_count_steps # (optional)
    save_checkpoints_steps = FLAGS.save_checkpoints_steps # (optional)
    keep_checkpoint_max = FLAGS.keep_checkpoint_max # (optional)
    data_dir = FLAGS.data_dir
    delimiter = FLAGS.delimiter
    separator  =FLAGS.separator
    batch_size = FLAGS.batch_size
    epochs = FLAGS.epochs
    field_size_user_profile = FLAGS.field_size_user_profile
    field_size_item_profile = FLAGS.field_size_item_profile
    field_size_context = FLAGS.field_size_context
    behaviors_size = FLAGS.behaviors_size
    shuffle = FLAGS.shuffle # (optional)
    buffer_size = FLAGS.buffer_size # (optional)
    num_parallel_calls = FLAGS.num_parallel_calls # (optional)
    use_dtype_high_precision = FLAGS.use_dtype_high_precision # (optional)
    name_feat_inds_user = FLAGS.name_feat_inds_user # (optional)
    name_feat_vals_user = FLAGS.name_feat_vals_user # (optional)
    name_feat_inds_item = FLAGS.name_feat_inds_item # (optional)
    name_feat_vals_item = FLAGS.name_feat_vals_item # (optional)
    name_feat_inds_context = FLAGS.name_feat_inds_context # (optional)
    name_feat_vals_context = FLAGS.name_feat_vals_context # (optional)
    name_feat_inds_candidate = FLAGS.name_feat_inds_candidate # (optional)
    name_feat_inds_behaviors = FLAGS.name_feat_inds_behaviors # (optional)
    task = FLAGS.task
    output_size = FLAGS.output_size
    feat_size_user_profile = FLAGS.feat_size_user_profile
    feat_size_item_profile = FLAGS.feat_size_item_profile
    feat_size_context = FLAGS.feat_size_context
    feat_size_id = FLAGS.feat_size_id
    embed_size_user_profile = FLAGS.embed_size_user_profile
    embed_size_item_profile = FLAGS.embed_size_item_profile
    embed_size_context = FLAGS.embed_size_context
    embed_size_id = FLAGS.embed_size_id
    hidden_sizes = FLAGS.hidden_sizes
    dropouts = FLAGS.dropouts
    use_softmax_norm_for_attention = FLAGS.use_softmax_norm_for_attention # (optional)
    use_bn = FLAGS.use_bn # (optional)
    use_global_bias = FLAGS.use_global_bias # (optional)
    use_hidden_bias = FLAGS.use_hidden_bias # (optional)
    lamb = FLAGS.lamb # (optional)
    optimizer = FLAGS.optimizer # (optional)
    learning_rate = FLAGS.learning_rate # (optional)

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
        "output_size": output_size,
        "field_size_user_profile": field_size_user_profile,
        "field_size_item_profile": field_size_item_profile,
        "field_size_context": field_size_context,
        "feat_size_user_profile": feat_size_user_profile,
        "feat_size_item_profile": feat_size_item_profile,
        "feat_size_context": feat_size_context,
        "feat_size_id": feat_size_id,
        "embed_size_user_profile": embed_size_user_profile,
        "embed_size_item_profile": embed_size_item_profile,
        "embed_size_context": embed_size_context,
        "embed_size_id": embed_size_id,
        "hidden_sizes": hidden_sizes,
        "dropouts": dropouts,
        "use_softmax_norm_for_attention": use_softmax_norm_for_attention,
        "use_bn": use_bn,
        "use_global_bias": use_global_bias,
        "use_hidden_bias": use_hidden_bias,
        "lamb": lamb,
        "optimizer": optimizer,
        "learning_rate": learning_rate,
        "dtype": dtype,
        "name_feat_inds_user": name_feat_inds_user,
        "name_feat_vals_user": name_feat_vals_user,
        "name_feat_inds_item": name_feat_inds_item,
        "name_feat_vals_item": name_feat_vals_item,
        "name_feat_inds_context": name_feat_inds_context,
        "name_feat_vals_context": name_feat_vals_context,
        "name_feat_inds_candidate": name_feat_inds_candidate,
        "name_feat_inds_behaviors": name_feat_inds_behaviors,
        "reuse": REUSE,
        "seed": SEED
    }

    # -------Multi-GPU Usage-------
    # mirrored_strategy = tfc.distribute.MirroredStrategy(devices=["/gpu:0", "/gpu:1"])
    # config = tf.estimator.RunConfig(
    #     log_step_count_steps=log_step_count_steps,
    #     save_checkpoints_steps=save_checkpoints_steps,
    #     keep_checkpoint_max=keep_checkpoint_max,
    #     train_distribute=mirrored_strategy,
    #     eval_distribute=mirrored_strategy
    # )

    config = tf.estimator.RunConfig(
        log_step_count_steps=log_step_count_steps,
        save_checkpoints_steps=save_checkpoints_steps,
        keep_checkpoint_max=keep_checkpoint_max
    )
    estimator = tf.estimator.Estimator(model_fn=model_fn, model_dir=model_dir, params=hparams, config=config)
    if phase == "train":
        filenames_train = tf.gfile.Glob(filename=os.path.join(data_dir, PREFIX_TRAIN_FILE + "*"))
        estimator.train(input_fn=lambda : input_fn(filenames=filenames_train,
                                                   delimiter=delimiter,
                                                   separator=separator,
                                                   batch_size=batch_size,
                                                   epochs=epochs,
                                                   field_size_user_profile=field_size_user_profile,
                                                   field_size_item_profile=field_size_item_profile,
                                                   field_size_context=field_size_context,
                                                   behaviors_size=behaviors_size,
                                                   shuffle=shuffle,
                                                   buffer_size=buffer_size,
                                                   num_parallel_calls=num_parallel_calls,
                                                   dtype=dtype,
                                                   name_feat_inds_user=name_feat_inds_user,
                                                   name_feat_vals_user=name_feat_vals_user,
                                                   name_feat_inds_item=name_feat_inds_item,
                                                   name_feat_vals_item=name_feat_vals_item,
                                                   name_feat_inds_context=name_feat_inds_context,
                                                   name_feat_vals_context=name_feat_vals_context,
                                                   name_feat_inds_candidate=name_feat_inds_candidate,
                                                   name_feat_inds_behaviors=name_feat_inds_behaviors),
                        steps=None,
                        max_steps=None)
    elif phase == "train-with-eval":
        filenames_train = tf.gfile.Glob(filename=os.path.join(data_dir, PREFIX_TRAIN_FILE + "*"))
        filenames_eval = tf.gfile.Glob(filename=os.path.join(data_dir, PREFIX_EVAL_FILE + "*"))
        train_spec = tf.estimator.TrainSpec(input_fn=lambda : input_fn(filenames=filenames_train,
                                                                       delimiter=delimiter,
                                                                       separator=separator,
                                                                       batch_size=batch_size,
                                                                       epochs=epochs,
                                                                       field_size_user_profile=field_size_user_profile,
                                                                       field_size_item_profile=field_size_item_profile,
                                                                       field_size_context=field_size_context,
                                                                       behaviors_size=behaviors_size,
                                                                       shuffle=shuffle,
                                                                       buffer_size=buffer_size,
                                                                       num_parallel_calls=num_parallel_calls,
                                                                       dtype=dtype,
                                                                       name_feat_inds_user=name_feat_inds_user,
                                                                       name_feat_vals_user=name_feat_vals_user,
                                                                       name_feat_inds_item=name_feat_inds_item,
                                                                       name_feat_vals_item=name_feat_vals_item,
                                                                       name_feat_inds_context=name_feat_inds_context,
                                                                       name_feat_vals_context=name_feat_vals_context,
                                                                       name_feat_inds_candidate=name_feat_inds_candidate,
                                                                       name_feat_inds_behaviors=name_feat_inds_behaviors),
                                            max_steps=None)
        eval_spec = tf.estimator.EvalSpec(input_fn=lambda : input_fn(filenames=filenames_eval,
                                                                     delimiter=delimiter,
                                                                     separator=separator,
                                                                     batch_size=batch_size,
                                                                     epochs=1,
                                                                     field_size_user_profile=field_size_user_profile,
                                                                     field_size_item_profile=field_size_item_profile,
                                                                     field_size_context=feat_size_context,
                                                                     behaviors_size=behaviors_size,
                                                                     shuffle=False,
                                                                     buffer_size=buffer_size,
                                                                     num_parallel_calls=num_parallel_calls,
                                                                     dtype=dtype,
                                                                     name_feat_inds_user=name_feat_inds_user,
                                                                     name_feat_vals_user=name_feat_vals_user,
                                                                     name_feat_inds_item=name_feat_inds_item,
                                                                     name_feat_vals_item=name_feat_vals_item,
                                                                     name_feat_inds_context=name_feat_inds_context,
                                                                     name_feat_vals_context=name_feat_vals_context,
                                                                     name_feat_inds_candidate=name_feat_inds_candidate,
                                                                     name_feat_inds_behaviors=name_feat_inds_behaviors),
                                          steps=None,
                                          start_delay_secs=120,
                                          throttle_secs=600)
        tf.estimator.train_and_evaluate(estimator=estimator, train_spec=train_spec, eval_spec=eval_spec)
    elif phase == "eval":
        filenames_eval = tf.gfile.Glob(filename=os.path.join(data_dir, PREFIX_EVAL_FILE + "*"))
        estimator.evaluate(input_fn=lambda :input_fn(filenames=filenames_eval,
                                                     delimiter=delimiter,
                                                     separator=separator,
                                                     batch_size=batch_size,
                                                     epochs=1,
                                                     field_size_user_profile=field_size_user_profile,
                                                     field_size_item_profile=field_size_item_profile,
                                                     field_size_context=feat_size_context,
                                                     behaviors_size=behaviors_size,
                                                     shuffle=False,
                                                     buffer_size=buffer_size,
                                                     num_parallel_calls=num_parallel_calls,
                                                     dtype=dtype,
                                                     name_feat_inds_user=name_feat_inds_user,
                                                     name_feat_vals_user=name_feat_vals_user,
                                                     name_feat_inds_item=name_feat_inds_item,
                                                     name_feat_vals_item=name_feat_vals_item,
                                                     name_feat_inds_context=name_feat_inds_context,
                                                     name_feat_vals_context=name_feat_vals_context,
                                                     name_feat_inds_candidate=name_feat_inds_candidate,
                                                     name_feat_inds_behaviors=name_feat_inds_behaviors))
    elif phase == "predict":
        filenames_predict = tf.gfile.Glob(filename=os.path.join(data_dir, PREFIX_PREDICT_FILE + "*"))
        P = estimator.predict(input_fn=lambda : input_fn(filenames=filenames_predict,
                                                         delimiter=delimiter,
                                                         separator=separator,
                                                         batch_size=batch_size,
                                                         epochs=1,
                                                         field_size_user_profile=field_size_user_profile,
                                                         field_size_item_profile=field_size_item_profile,
                                                         field_size_context=feat_size_context,
                                                         behaviors_size=behaviors_size,
                                                         shuffle=False,
                                                         buffer_size=buffer_size,
                                                         num_parallel_calls=num_parallel_calls,
                                                         dtype=dtype,
                                                         name_feat_inds_user=name_feat_inds_user,
                                                         name_feat_vals_user=name_feat_vals_user,
                                                         name_feat_inds_item=name_feat_inds_item,
                                                         name_feat_vals_item=name_feat_vals_item,
                                                         name_feat_inds_context=name_feat_inds_context,
                                                         name_feat_vals_context=name_feat_vals_context,
                                                         name_feat_inds_candidate=name_feat_inds_candidate,
                                                         name_feat_inds_behaviors=name_feat_inds_behaviors))
        for p in P:
            print(p)
    elif phase == "export":
        features = {
            name_feat_inds_user: tf.placeholder(dtype=tf.int32, shape=[None, field_size_user_profile], name=name_feat_inds_user),
            name_feat_vals_user: tf.placeholder(dtype=dtype, shape=[None, field_size_user_profile], name=name_feat_vals_user),
            name_feat_inds_item: tf.placeholder(dtype=tf.int32, shape=[None, field_size_item_profile], name=name_feat_inds_item),
            name_feat_vals_item: tf.placeholder(dtype=dtype, shape=[None, field_size_item_profile], name=name_feat_vals_item),
            name_feat_inds_context: tf.placeholder(dtype=dtype, shape=[None, field_size_context], name=name_feat_inds_context),
            name_feat_vals_context: tf.placeholder(dtype=dtype, shape=[None, field_size_context], name=name_feat_vals_context),
            name_feat_inds_candidate: tf.placeholder(dtype=tf.int32, shape=[None], name=name_feat_inds_candidate),
            name_feat_inds_behaviors: tf.placeholder(dtype=tf.int32, shape=[None, behaviors_size], name=name_feat_inds_behaviors)
        }
        serving_input_receiver_fn = tf.estimator.export.build_raw_serving_input_receiver_fn(features=features)
        estimator.export_savedmodel(export_dir_base=export_dir, serving_input_receiver_fn=serving_input_receiver_fn)
    else:
        raise NotImplementedError("Argument <phase> value: {} is not supported.".format(phase))


if __name__ == '__main__':
    tf.logging.set_verbosity(v=tf.logging.INFO)
    tf.app.run(main=main)