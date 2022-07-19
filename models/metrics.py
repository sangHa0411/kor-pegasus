
import tensorflow as tf

class CategoricalAccuracy(tf.keras.metrics.Metric):

    def __init__(self, name='categorical accuracy', **kwargs):
        super(CategoricalAccuracy, self).__init__(name=name, **kwargs)
        self.acc = self.add_weight(name='acc', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.squeeze(tf.cast(y_true, tf.int32), axis=1)
        y_pred = tf.cast(y_pred, tf.float32)

        y_pred_arg = tf.cast(tf.math.argmax(y_pred, axis=1), tf.int32)
        values = tf.cast(tf.equal(y_true, y_pred_arg), tf.float32)
        self.acc.assign(tf.reduce_mean(values))

    def result(self):
        return self.acc

    def reset_states(self):
        self.acc.assign(0)