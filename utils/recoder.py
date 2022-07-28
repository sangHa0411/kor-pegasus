
import tensorflow as tf
from tqdm import tqdm

class Recoder :

    def __init__(self, max_input_length, max_target_length) :
        self.max_input_length = max_input_length
        self,max_target_length = max_target_length


    def _int64_feature(self, values):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


    def serialize_example(self, data):
        input_ids = data["input_ids"]
        attention_mask = data["attention_mask"]
        decoder_input_ids = data["decoder_input_ids"]
        decoder_attention_mask = data["decoder_attention_mask"]
        labels = data["labels"]

        feature = {
            "input_ids": self._int64_feature(input_ids),
            "attention_mask": self._int64_feature(attention_mask),
            "decoder_input_ids": self._int64_feature(decoder_input_ids),
            "decoder_attention_mask": self._int64_feature(decoder_attention_mask),
            "labels": self._int64_feature(labels)
        }

        example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
        return example_proto.SerializeToString()


    def write(self, dataset, path) :
        assert path.endswith(".tfrecord")
        with tf.io.TFRecordWriter(path) as writer:
            for d in tqdm(dataset):
                example = self.serialize_example(d)
                writer.write(example)


    def read(self, path, batch_size) :
        raw_dataset = tf.data.TFRecordDataset(path)

        # Create a description of the features.
        INPUT_SIZE = self.max_input_length
        OUTPUT_SIZE = self.max_target_length
        feature_description = {
            "input_ids": tf.io.FixedLenFeature([INPUT_SIZE], tf.int64),
            'attention_mask': tf.io.FixedLenFeature([INPUT_SIZE], tf.int64),
            "decoder_input_ids": tf.io.FixedLenFeature([OUTPUT_SIZE], tf.int64),
            "decoder_attention_mask": tf.io.FixedLenFeature([OUTPUT_SIZE], tf.int64),
            "labels": tf.io.FixedLenFeature([OUTPUT_SIZE], tf.int64),
        }

        def _parse_function(example_proto):
            # Parse the input `tf.train.Example` proto using the dictionary above.
            return tf.io.parse_single_example(example_proto, feature_description)

        tf_dataset = raw_dataset.map(_parse_function).batch(batch_size)
        return tf_dataset