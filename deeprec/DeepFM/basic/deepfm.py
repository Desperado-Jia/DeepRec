"""
Created on 2019/04/12 19:22
author: Tong Jia
email: cecilio.jia@gmail.com
Description:
    An implementation of DeepFM: A Factorization-Machine based Neural Network for CTR Prediction (DeepFM).
    See algorithm and hyperparameter details:
        [Guo et al., 2017](https://www.ijcai.org/proceedings/2017/0239.pdf)
    The algorithm is developed based on TensorFlow 1.12.0 version.
"""
import tensorflow as tf

def data_generator(filenames,
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
    def map_func(line):
        columns = tf.string_split(source=[line], delimiter=delimiter, skip_empty=False)
        label = tf.string_to_number(string_tensor=columns[0], out_type=dtype)
        splits = tf.string_split(source=columns[1: ], delimiter=separator, skip_empty=False)
        feats = tf.reshape(
            tensor=splits.values, shape=splits.dense_shape
        ) # A tensor in shape of (field_size, 2), 1-th column contains feature indices, 2-th column contains feature values
        inds, vals = tf.split(value=feats, num_or_size_splits=2, axis=1) # Two tensors in shape of (field_size, 1)
        inds = tf.squeeze(input=tf.string_to_number(string_tensor=inds, out_type=tf.int32),
                          axis=-1) # A tensor in shape of (field_size)
        vals = tf.squeeze(input=tf.string_to_number(string_tensor=vals, out_type=dtype),
                          axis=-1) # A tensor in shape of (field_size)
        feat_dict = {name_feat_inds: inds, name_feat_vals: vals}
        return feat_dict, label
    dataset = tf.data.TextLineDataset(filenames=filenames, buffer_size=buffer_size).\
        map(map_func=map_func, num_parallel_calls=num_parallel_calls)
    if shuffle == True:
        dataset = dataset.shuffle(buffer_size=10 * batch_size)
    dataset = dataset. \
        repeat(count=epochs). \
        batch(batch_size=batch_size, drop_remainder=False)
    dataset = dataset.prefetch(buffer_size=1)
    iterator = dataset.make_one_shot_iterator()
    return iterator

if __name__ == '__main__':
