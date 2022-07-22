
import tensorflow as tf
import tensorflow_addons as tfa
from models.scheduler import LinearWarmupSchedule
from models.loss import SparseCategoricalCrossentropy
from models.metrics import Accuracy

class Trainer :

    def __init__(self, args, model_create_fn, tokenizer, datasets, tpu_name) :
        self.args = args
        self.model_create_fn = model_create_fn
        self.tokenizer = tokenizer
        self.datasets = datasets
        self.tpu_name = tpu_name


    def get_optimizer(self,) :
        steps_per_epoch = int(len(self.datasets) / self.args.batch_size)
        total_steps = steps_per_epoch * self.args.epochs
        warmup_scheduler = LinearWarmupSchedule(total_steps, 
            self.args.warmup_ratio, 
            self.args.learning_rate
        )
        optimizer = tfa.optimizers.AdamW(learning_rate=warmup_scheduler, weight_decay=self.args.weight_decay)

        return optimizer

    def get_tf_datasets(self, datasets, batch_size) :
        input_ids = datasets["input_ids"]
        attention_mask = datasets["attention_mask"]
        decoder_input_ids = datasets["decoder_input_ids"]
        decoder_attention_mask = datasets["decoder_attention_mask"]

        labels = datasets["labels"]
        
        input_tensors = tf.data.Dataset.from_tensor_slices((input_ids, 
            attention_mask, 
            decoder_input_ids, 
            decoder_attention_mask)
            ).batch(batch_size)
        label_tensors = tf.data.Dataset.from_tensor_slices(labels).batch(batch_size)

        tf_datasets = tf.data.Dataset.zip((input_tensors, label_tensors))
        return tf_datasets

    def train_keras(self) :
        # -- Setting TPU
        resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu=self.tpu_name)
        tf.config.experimental_connect_to_cluster(resolver)
        tf.tpu.experimental.initialize_tpu_system(resolver)

        # -- Training
        strategy = tf.distribute.TPUStrategy(resolver)
        tf_datasets = self.get_tf_datasets(self.datasets, self.args.batch_size)

        with strategy.scope() :
            model = self.model_create_fn()

            optimizer = self.get_optimizer()
            model.compile(optimizer=optimizer, 
                loss=SparseCategoricalCrossentropy(self.tokenizer),
                metrics=[Accuracy(self.tokenizer)]
            )

        model.fit(tf_datasets, epochs=self.args.epochs, verbose=1)

    def train(self) :
        # -- Setting TPU
        resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu=self.tpu_name)
        tf.config.experimental_connect_to_cluster(resolver)
        tf.tpu.experimental.initialize_tpu_system(resolver)

        # -- Training
        strategy = tf.distribute.TPUStrategy(resolver)

        with strategy.scope():
            model = self.model_create_fn()
            optimizer = self.get_optimizer()

            # loss_fn = SparseCategoricalCrossentropy(self.tokenizer)
            training_loss = tf.keras.metrics.Mean('training_loss', dtype=tf.float32)
            training_accuracy = Accuracy("training_acc", self.tokenizer)

        per_replica_batch_size = self.args.batch_size // strategy.num_replicas_in_sync
        tf_datasets = strategy.distribute_datasets_from_function(
            lambda _: self.get_tf_datasets(self.datasets, batch_size=per_replica_batch_size)
        )

        @tf.function
        def train_step(iterator):
            """The step function for one training step."""
            def step_fn(data):
                inputs, labels = data
                with tf.GradientTape() as tape:
                    logits = model(inputs, training=True)
                    loss = tf.keras.losses.sparse_categorical_crossentropy(
                        labels, logits, from_logits=True)

                    loss = tf.where(labels==self.tokenizer.pad_token_id, 0.0, loss)
                    loss = tf.reduce_mean(loss)
                
                grads = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(list(zip(grads, model.trainable_variables)))

                training_loss.update_state(loss * strategy.num_replicas_in_sync)
                training_accuracy.update_state(labels, logits)

            strategy.run(step_fn, args=(next(iterator),))

        steps_per_epoch = int(len(self.datasets) / self.args.batch_size)
        train_iterator = iter(tf_datasets)
        for epoch in range(self.args.epochs):
            print('Epoch: {}/{}'.format(epoch, self.args.epochs))

            for step in range(steps_per_epoch):
                train_step(train_iterator)

                cur_step = optimizer.iterations.numpy()
                step_loss = round(float(training_loss.result()),4)
                step_acc = round(float(training_accuracy.result()),4)

                print('Current step: {}, training loss: {}, accuracy: {}%'.format(
                    cur_step, step_loss, step_acc)
                )

            training_loss.reset_states()
            training_accuracy.reset_states()