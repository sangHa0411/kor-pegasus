
import os
import sys
import torch
import random
import argparse
import pandas as pd
import numpy as np
import multiprocessing
import tensorflow as tf
import tensorflow_addons as tfa

from utils.loader import DataLoader
from utils.preprocessor import Preprocessor
from utils.encoder import Encoder
from utils.collator import DataCollatorForSeq2Seq

from models.scheduler import LinearWarmupSchedule

import wandb
from dotenv import load_dotenv

from arguments import ModelArguments, DataArguments, TrainingArguments, LoggingArguments
from transformers import (
    PegasusConfig,
    PegasusTokenizerFast,
    TFPegasusForConditionalGeneration,
    HfArgumentParser
)

def main():

    parser = HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, LoggingArguments)
    )
    model_args, data_args, training_args, logging_args = parser.parse_args_into_dataclasses()
    seed_everything(training_args.seed)

    CPU_COUNT = multiprocessing.cpu_count() // 2

    # -- Loading datasets
    print('\nLoading Datasets')
    data_loader = DataLoader(seed=training_args.seed)
    datasets = data_loader.load(data_args.dir_path)
    print(datasets)

    # -- Preprocessing datasets
    print('\Preprocessing Datasets')
    preprocessor = Preprocessor()
    datasets = datasets.map(preprocessor, batched=True, num_proc=CPU_COUNT)
    print(datasets)

    # -- Encoding datasets
    print('\Encoding Datasets')
    tokenizer = PegasusTokenizerFast.from_pretrained(model_args.PLM, use_fast=True)
    encoder = Encoder(tokenizer, data_args.max_input_len, data_args.max_target_len)
    datasets = datasets.map(encoder, 
        batched=True,
        num_proc=CPU_COUNT,
        remove_columns = datasets.column_names
    )
    print(datasets)

    # -- Data Collator
    data_collator = DataCollatorForSeq2Seq(tokenizer, padding=True, max_length=data_args.max_input_len, return_tensors="tf")

    # -- Converting datasets type
    tf_datasets = datasets.to_tf_dataset(
        columns=["attention_mask", "input_ids"],
        label_cols=["labels"],
        shuffle=True,
        collate_fn=data_collator,
        batch_size=training_args.train_batch_size,
    )

    # -- Configuration
    config = PegasusConfig.from_pretrained(model_args.PLM)

    # -- Model
    print('\nLoading Model')
    def create_model():
        input_ids = tf.keras.layers.Input(shape=(data_args.max_input_len,), dtype=tf.int32, name="input_ids")
        attention_mask = tf.keras.layers.Input(shape=(data_args.max_input_len,), dtype=tf.int32, name="attention_mask")
        decoder_input_ids = tf.keras.layers.Input(shape=(args.max_len,), dtype=tf.int32, name="attention_mask")

        summarization_model = TFPegasusForConditionalGeneration.from_pretrained(model_args.PLM, config=config, from_pt=True)
        outputs = summarization_model(input_ids=input_ids, attention_mask = attention_mask, decoder_input_ids=decoder_input_ids)

        model = tf.keras.Model(inputs=[input_ids, attention_mask], outputs=outputs)
        return model
   
    # -- Setting TPU
    resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='tpu')
    tf.config.experimental_connect_to_cluster(resolver)
    tf.tpu.experimental.initialize_tpu_system(resolver)

    # -- Checking TPU Devices
    for i, cf in enumerate(tf.config.list_logical_devices('TPU')) :
        print("%dth devices: %s" %(i, cf))

    # -- Optmizer & Scheduler
    total_steps = len(tf_datasets) * training_args.epochs
    warmup_ratio = training_args.warmup_ratio
    warmup_scheduler = LinearWarmupSchedule(total_steps, warmup_ratio, training_args.learning_rate)
    optimizer = tfa.optimizers.AdamW(learning_rate=warmup_scheduler, weight_decay=training_args.weight_decay)

    # -- Training
    strategy = tf.distribute.TPUStrategy(resolver)

    with strategy.scope() :
        model = create_model()
        model.compile(optimizer=optimizer,
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            # metrics=["accuracy"]
        )

    model.fit(tf_datasets, 
        epochs=training_args.epochs, 
        verbose=1,
    )

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

if __name__ == '__main__':
    main()