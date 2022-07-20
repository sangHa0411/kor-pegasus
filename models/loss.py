
import tensorflow as tf

class SparseCategoricalCrossentropy :
    def __init__(self, tokenizer, label_pad_token_id) :
        self.tokenizer = tokenizer
        self.label_pad_token_id = label_pad_token_id
        self.loss = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True,
            reduction=tf.keras.losses.Reduction.NONE
        )

    def __call__(self, y_true, y_pred) :
        y_true = tf.cast(y_true, tf.int32)
        y_true = tf.where(y_true == self.label_pad_token_id, self.tokenizer.pad_token_id, y_true)

        loss = self.loss(y_true, y_pred)
        loss = tf.where(y_true == self.tokenizer.pad_token_id, 0.0, loss)
        
        loss_per_example = tf.reduce_mean(loss, axis=-1)
        loss_mean = tf.reduce_mean(loss_per_example)
        return loss_mean