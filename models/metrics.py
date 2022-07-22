
import tensorflow as tf

class Accuracy(tf.keras.metrics.Metric) :

    def __init__(self, tokenizer) :
        super(Accuracy, self).__init__()
        self.tokenizer = tokenizer
        self.acc = self.add_weight(name='acc', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None) :
        y_true = tf.cast(y_true, tf.int32)
        
        y_pred_arg = tf.cast(tf.math.argmax(y_pred, axis=-1), tf.int32)
        total_tokens = tf.where(y_true == self.tokenizer.pad_token_id, 0.0, 1.0)
        total_size = tf.reduce_sum(total_tokens)

        correct_tokens = tf.where(y_true == y_pred_arg, 1.0, 0.0)
        correct_tokens = tf.where(y_true == self.tokenizer.pad_token_id, 0.0, correct_tokens)
        correct_size = tf.reduce_sum(correct_tokens)

        self.acc.assign_add(correct_size / total_size)

    def result(self) :
        return self.acc