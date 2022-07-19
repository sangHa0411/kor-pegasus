
import tensorflow as tf

class LinearWarmupSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
	def __init__(self, total_steps, warmup_ratio, learning_rate, name=None):
		self.total_steps = total_steps	
		self.warmup_ratio = warmup_ratio
		self.learning_rate = learning_rate
		self.name = name

	def __call__(self, step):
		total_steps = self.total_steps + 1
		warmup_steps = int(self.warmup_ratio * total_steps)

		with tf.name_scope(self.name or "LinearWarmupSchedule") as name:
			learning_rate = tf.convert_to_tensor(
				self.learning_rate, name="learning_rate"
			)

			dtype = learning_rate.dtype
			total_steps = tf.cast(total_steps, dtype)
			warmup_steps = tf.cast(warmup_steps, dtype)

			flag = tf.less(step, warmup_steps)
			current_lr = tf.cond(flag, 
				true_fn = lambda: learning_rate * (step / warmup_steps), 
				false_fn = lambda: learning_rate * 
					(1 - (step - warmup_steps) / (total_steps - warmup_steps)),
				name=name
			)

			return current_lr

	def get_config(self):
		return {
			"learning_rate": self.learning_rate,
			"warmup_ratio": self.warmup_ratio,
			"total_steps": self.total_steps,
			"name": self.name
		}