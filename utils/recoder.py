
import os
import tensorflow as tf
from tqdm import tqdm

class Recoder :

    def __init__(self, bucket, dir_path, max_input_length, max_target_length) :
        self.bucket = bucket
        self.dir_path = dir_path
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length


    def _int64_feature(self, values):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


    def serialize_data(self, data):
        input_ids = data["input_ids"]
        attention_mask = data["attention_mask"]
        decoder_input_ids = data["decoder_input_ids"]
        decoder_attention_mask = data["decoder_attention_mask"]

        feature = {
            "input_ids": self._int64_feature(input_ids),
            "attention_mask": self._int64_feature(attention_mask),
            "decoder_input_ids": self._int64_feature(decoder_input_ids),
            "decoder_attention_mask": self._int64_feature(decoder_attention_mask),
        }

        example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
        return example_proto.SerializeToString()

    def serialize_label(self, data):
        labels = data["labels"]

        feature = {
            "labels": self._int64_feature(labels)
        }

        example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
        return example_proto.SerializeToString()

    # not working
    def write(self, dataset) :
        storage_name = "gs://two-ai"
        storage_path = os.path.join(storage_name, "dataset")
        path = os.path.join(storage_path, "dataset.tfrecord")

        with tf.io.TFRecordWriter(path) as writer:
            for d in tqdm(dataset):
                example = self.serialize_data(d)
                writer.write(example)

        path = os.path.join(storage_path, "labels.tfrecord")
        with tf.io.TFRecordWriter(path) as writer:
            for d in tqdm(dataset):
                example = self.serialize_label(d)
                writer.write(example)

    def read(self, batch_size) :

        # Create a description of the features.
        INPUT_SIZE = self.max_input_length
        OUTPUT_SIZE = self.max_target_length
        feature_description = {
            "input_ids": tf.io.FixedLenFeature([INPUT_SIZE], tf.int64),
            'attention_mask': tf.io.FixedLenFeature([INPUT_SIZE], tf.int64),
            "decoder_input_ids": tf.io.FixedLenFeature([OUTPUT_SIZE], tf.int64),
            "decoder_attention_mask": tf.io.FixedLenFeature([OUTPUT_SIZE], tf.int64),
        }

        label_description = {
            "labels": tf.io.FixedLenFeature([OUTPUT_SIZE], tf.int64),
        }

        def _parse_feature_function(example_proto):
            return tf.io.parse_single_example(example_proto, feature_description)

        def _parse_label_function(example_proto):
            return tf.io.parse_single_example(example_proto, label_description)

        storage_name = "gs://two-ai"
        storage_path = os.path.join(storage_name, "dataset")

        raw_dataset = tf.data.TFRecordDataset(os.path.join(storage_path, "dataset.tfrecord"))
        tf_dataset = raw_dataset.map(_parse_feature_function).batch(batch_size)

        raw_labels = tf.data.TFRecordDataset(os.path.join(storage_path, "labels.tfrecord"))
        tf_labels = raw_labels.map(_parse_label_function).batch(batch_size)

        tf_zip_dataset = tf.data.Dataset.zip((tf_dataset, tf_labels))
        return tf_zip_dataset