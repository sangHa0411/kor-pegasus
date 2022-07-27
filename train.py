import torch
import random
import numpy as np
import tensorflow as tf

from datasets import load_dataset
from utils.loader import DataLoader
from utils.preprocessor import Preprocessor
from utils.encoder import Encoder
from trainer import Trainer

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

    CPU_COUNT = 8

    # -- Loading datasets
    # data_loader = DataLoader(seed=training_args.seed)
    # dataset = data_loader.load(data_args.dir_path)
    # dataset.push_to_hub("sh110495/summarization", private=True)
    dataset = load_dataset("sh110495/summarization", 
        use_auth_token=True, 
        split="train"
    )

    # -- Preprocessing datasets
    preprocessor = Preprocessor()
    dataset = dataset.map(preprocessor, batched=True, num_proc=CPU_COUNT)
    
    # -- Encoding datasets
    tokenizer = PegasusTokenizerFast.from_pretrained(model_args.PLM)
    encoder = Encoder(tokenizer, data_args.max_input_length, data_args.max_target_length)
    dataset = dataset.map(encoder, 
        batched=True, 
        num_proc=CPU_COUNT, 
        remove_columns=dataset.column_names
    )

    # -- Configuration
    config = PegasusConfig.from_pretrained(model_args.PLM)

    # -- Model
    print('\nLoading Model')
    def create_model():
        ## -- Model Inputs
        input_ids = tf.keras.layers.Input(shape=(data_args.max_input_length,), dtype=tf.int32, name="input_ids")
        attention_mask = tf.keras.layers.Input(shape=(data_args.max_input_length,), dtype=tf.int32, name="attention_mask")
        decoder_input_ids = tf.keras.layers.Input(shape=(data_args.max_target_length,), dtype=tf.int32, name="decoder_input_ids")
        decoder_attention_mask = tf.keras.layers.Input(shape=(data_args.max_target_length,), dtype=tf.int32, name="decoder_attention_mask")

        ## -- Model
        summarization_model = TFPegasusForConditionalGeneration.from_pretrained(model_args.PLM, config=config, from_pt=True)

        ## -- Model Outputs
        outputs = summarization_model(input_ids=input_ids, 
            attention_mask = attention_mask, 
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask
        )["logits"]

        ## -- Keras Model
        model = tf.keras.Model(inputs=[input_ids, attention_mask, decoder_input_ids, decoder_attention_mask], outputs=outputs)
        return model
   
    # -- Trainer
    trainer = Trainer(
        args=training_args,
        logging_args=logging_args,
        model_create_fn=create_model,
        tokenizer=tokenizer,
        datasets=dataset,
        tpu_name="tpu"
    )

    # -- Training
    trainer.train()

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

if __name__ == "__main__" :
    main()