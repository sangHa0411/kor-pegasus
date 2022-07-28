import os
import wandb
import tensorflow as tf
import tensorflow_addons as tfa
from tqdm import tqdm
from utils.recoder import Recoder
from dotenv import load_dotenv
from models.metrics import Accuracy
from models.scheduler import LinearWarmupSchedule


class Trainer :

    def __init__(self, args, data_args, logging_args, model_create_fn, tokenizer, datasets, tpu_name) :
        self.args = args
        self.logging_args = logging_args
        self.data_args = data_args
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


    def train(self) :

        recoder = Recoder(batch_size=self.args.batch_size, 
            max_input_length=self.data_args.max_input_length,
            max_target_length=self.data_args.max_target_length
        )

        record_path = os.path.join(self.data_args.dir_path, "dataset.tfrecord")
        recoder.write(dataset=self.dataset, path=record_path)

        # -- Setting TPU
        resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu=self.tpu_name)
        tf.config.experimental_connect_to_cluster(resolver)
        tf.tpu.experimental.initialize_tpu_system(resolver)

        # -- Strategy
        strategy = tf.distribute.TPUStrategy(resolver)

        with strategy.scope():
            model = self.model_create_fn()
            optimizer = self.get_optimizer()

            training_loss = tf.keras.metrics.Mean('training_loss', dtype=tf.float32)
            training_accuracy = Accuracy("training_acc", self.tokenizer)

        per_replica_batch_size = self.args.batch_size // strategy.num_replicas_in_sync
        tf_datasets = strategy.distribute_datasets_from_function(
            lambda _: recoder.read(record_path, per_replica_batch_size)
        )

        TPU_NUM = len(tf.config.list_logical_devices('TPU'))

        # -- Training function
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
                    loss = tf.reduce_mean(loss) / TPU_NUM
                
                grads = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(list(zip(grads, model.trainable_variables)))

                training_loss.update_state(loss * strategy.num_replicas_in_sync)
                training_accuracy.update_state(labels, logits)

            strategy.run(step_fn, args=(next(iterator),))

        # -- Training Steps
        steps_per_epoch = int(len(self.datasets) / self.args.batch_size)
        total_steps = steps_per_epoch * self.args.epochs

        train_iterator = iter(tf_datasets)

        # -- Wandb Setting
        load_dotenv(dotenv_path=self.logging_args.dotenv_path)
        WANDB_AUTH_KEY = os.getenv('WANDB_AUTH_KEY')
        wandb.login(key=WANDB_AUTH_KEY)

        wandb.init(entity="sangha0411", 
            project=self.logging_args.project_name,
            name=self.logging_args.run_name,
            group=self.logging_args.group_name,
        )
        wandb.config.update(self.args)

        # -- Training loop
        print("\nTraining")
        print("The number of data : %d" %len(self.datasets))
        print("Training Batch Size : %d" %self.args.batch_size)
        print("Total Steps : %d" %total_steps)
        print("The number of training epochs : %d" %self.args.epochs)
        print("Training steps per epoch : %d" %steps_per_epoch)
        print("The number of TPU cores : %d" %TPU_NUM)
        
        progress_bar = tqdm(range(total_steps))
        for step in progress_bar :
            progress_bar.set_description("{}/{}".format(step, total_steps))  
            train_step(train_iterator)

            cur_step = optimizer.iterations.numpy()
            # -- Logging to wandb
            if cur_step % self.args.logging_steps == 0 :
                cur_info = {
                    "loss" : round(float(training_loss.result()), 4),
                    "step" : cur_step ,
                    "accuracy" : round(float(training_accuracy.result()), 4),
                }
                self.log(cur_info)

            # -- Save checkpoint
            if cur_step % self.args.save_steps == 0 :
                training_loss.reset_states()
                training_accuracy.reset_states()

                self.save_model(model, cur_step)
        
        # -- Save final checkpoint
        self.save_model(model, cur_step)
        wandb.finish()


    def save_model(self, model, cur_step) :
        checkpoints = os.listdir(self.args.save_path)

        LIMIT = self.args.save_total_limit
        if len(checkpoints) > LIMIT :
            checkpoints = sorted(checkpoints)
            target = os.path.join(self.args.save_path, checkpoints[0])

            print("\nRemoving {}".format(target))

        save_path = os.path.join(self.args.save_path, "checkpoints-{}.h5".format(cur_step))
        print("Saving {}\n".format(save_path))
        model.save(save_path)


    def log(self, info) :
        cur_step = info["step"]
        cur_loss = info["loss"]
        cur_accuracy = info["accuracy"]

        print("Current step: {}, training loss: {}, accuracy: {}%".format(
            cur_step, cur_loss, cur_accuracy)
        )

        wandb.log({"train/loss" : cur_loss, 
            "train/accuracy" : cur_accuracy}, 
            step=cur_step
        )