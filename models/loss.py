import tensorflow as tf

class SparseCategoricalCrossentropy(tf.keras.losses.Loss) :
    def __init__(self, tokenizer) :
        super().__init__()
        self.loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True,
            reduction=tf.keras.losses.Reduction.NONE
        )
        self.tokenizer = tokenizer

    def call(self, y_true, y_pred) :
        y_loss = self.loss(y_true, y_pred)
        y_loss = tf.where(y_true == self.tokenizer.pad_token_id, 0.0, y_loss)
        loss_per_example = tf.reduce_mean(y_loss, axis=1)
        return tf.reduce_mean(loss_per_example)
