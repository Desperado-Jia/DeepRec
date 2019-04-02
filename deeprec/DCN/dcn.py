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
        if field_size_numerical != 0:
            # ----------Process numerical fields----------
            vals_numerical = tf.string_to_number(string_tensor=columns[1: 1 + field_size_numerical],
                                                 out_type=dtype) # A tensor in shape of (field_size_numerical)
            # ----------Process categorical fields----------
            inds_categorical = tf.string_to_number(string_tensor=columns[1 + field_size_numerical: ],
                                                   out_type=tf.int32) # A tensor in shape of (field_size_categorical)
            feats = {name_feat_vals_numerical: vals_numerical,
                        name_feat_inds_categorical: inds_categorical}
        else:
            # ----------Process categorical fields----------
            inds_categorical = tf.string_to_number(string_tensor=columns[1: ],
                                                   out_type=tf.int32) # A tensor in shape of (field_size_categorical)
            feats = {name_feat_inds_categorical: inds_categorical}
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



if __name__ == '__main__':
    features, labels = input_fn(filenames=["data.txt"],
                                delimiter=" ",
                                field_size_numerical=0,
                                batch_size=2,
                                epochs=10,
                                shuffle=False)
    with tf.Session() as sess:
        f = sess.run(fetches=features)