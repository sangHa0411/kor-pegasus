
import tensorflow as tf

def get_accuracy(y_true, y_pred):
    y_true = tf.cast(y_true, tf.int32)
    y_pred_arg = tf.cast(tf.math.argmax(y_pred, axis=-1), tf.int32)

    total_tokens = tf.where(y_true == -100, 0.0, 1.0)
    total_size = tf.reduce_sum(total_tokens)

    correct_tokens = tf.where(y_true == y_pred_arg, 1.0, 0.0)
    correct_tokens = tf.where(y_true == -100, 0.0, correct_tokens)
    correct_size = tf.reduce_sum(correct_tokens)

    acc = correct_size / total_size
    return acc