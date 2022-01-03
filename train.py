
import os
import sys
import argparse
import pandas as pd
import numpy as np
import random
import torch

from loader import DataLoader
from preprocessor import Masking, Preprocessor
from collator import DataCollatorForPegasus

import wandb
from dotenv import load_dotenv

from model import PegasusForPretraining
from transformers.trainer_utils import get_last_checkpoint
from transformers import (
    AutoConfig,
    AutoTokenizer,
    Seq2SeqTrainingArguments, 
    Seq2SeqTrainer
)

def train(args):

    # -- Device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # -- Checkpoint
    model_checkpoint = args.PLM
    
    # -- Datasets
    print('\nLoading Article Data')
    api_key = os.getenv('HUGGINGFACE_AUTH_KEY')
    article_loader = DataLoader(data_size=10, seed=args.seed)
    datasets = article_loader.load_data(api_key=api_key)
    print(datasets)

    # -- Preprocessing
    print('\nPreprocessing Data')
    masking = Masking(0.3)
    datasets = datasets.map(masking, 
        batched=True,
        num_proc=4,
        load_from_cache_file=True,
    )
    print(datasets)
    
    # -- Tokenizing
    print('\nTokenizing')
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)
    preprocessor = Preprocessor(tokenizer, args.max_input_len, args.max_target_len)
    datasets = datasets.map(preprocessor, 
        batched=True,
        num_proc=4,
        load_from_cache_file=True,
        remove_columns = datasets.column_names
    )
    print(datasets)

    # -- Configuration
    config = AutoConfig.from_pretrained(model_checkpoint)
    print(config)

    # -- Model
    print('\nLoading Model')
    model = PegasusForPretraining.from_pretrained(model_checkpoint, config=config).to(device)
    print('Model Type : {}'.format(type(model)))

    training_args = Seq2SeqTrainingArguments(
        output_dir = args.output_dir,                                   # output directory
        logging_dir = args.logging_dir,                                 # logging directory
        num_train_epochs = args.epochs,                                 # epochs
        save_strategy = 'epoch',                                        # save strategy
        logging_strategy = 'steps',                                     # logging strategy
        logging_steps = 1000,                                           # logging steps                                         
        per_device_train_batch_size = args.train_batch_size,            # train batch size
        warmup_steps=args.warmup_steps,                                 # warmup steps
        weight_decay=args.weight_decay,                                 # weight decay
        learning_rate = args.learning_rate,                             # learning rate
        gradient_accumulation_steps=args.gradient_accumulation_steps,   # accumulation steps
        fp16=True if args.fp16 == 1 else False,                         # fp 16 flag
        overwrite_output_dir=True if args.overwrite_output_dir == 1 else False
    )
 
    # -- Collator
    data_collator = DataCollatorForPegasus(tokenizer, mlm=True, mlm_probability=0.15)

    # -- Trainer
    trainer = Seq2SeqTrainer(
        model,
        training_args,
        train_dataset=datasets,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    # -- Training
    print('\nTraining')
    last_checkpoint = None
    if (
        os.path.isdir(args.output_dir)
        and not args.overwrite_output_dir
    ):

        last_checkpoint = get_last_checkpoint(args.output_dir)
        if last_checkpoint is None and len(os.listdir(args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )

        train_result = trainer.train(resume_from_checkpoint=last_checkpoint)
        print("Train result: ", train_result)
        trainer.save_model()

def main(args):
    load_dotenv(dotenv_path=args.dotenv_path)
    WANDB_AUTH_KEY = os.getenv('WANDB_AUTH_KEY')
    wandb.login(key=WANDB_AUTH_KEY)

    wandb_name = f"epochs:{args.epochs}_batch_size:{args.train_batch_size}_warmup_steps:{args.warmup_steps}_weight_decay:{args.weight_decay}"
    wandb.init(
        entity="sangha0411",
        project="PEGASUS pretrainging", 
        name=wandb_name,
        group='pegasus')

    wandb.config.update(args)
    train(args)
    wandb.finish()

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # -- Directory
    parser.add_argument('--output_dir', default='./results', help='model save at {SM_SAVE_DIR}/{name}')
    parser.add_argument('--logging_dir', default='./logs', help='logging save at {SM_SAVE_DIR}/{name}')

    # -- Model
    parser.add_argument('--PLM', type=str, default='sh110495/kor-pegasus', help='model type (default: sh110495/kor-pegasus)')

    # -- Training
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train (default: 10)')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='learning rate (default: 1e-4)')
    parser.add_argument('--train_batch_size', type=int, default=2, help='train batch size (default: 2)')
    parser.add_argument('--warmup_steps', type=int, default=20000, help='number of warmup steps for learning rate scheduler (default: 20000)')
    parser.add_argument('--weight_decay', type=float, default=1e-2, help='streng1th of weight decay (default: 1e-2)')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=16, help='gradient_accumulation_steps of training (default: 16)')
    parser.add_argument('--fp16', type=int, default=0, help='using fp16 (default: 0)')
    parser.add_argument('--overwrite_output_dir', type=int, default=0, help='overwriting output directory')

    # -- Data
    parser.add_argument('--max_input_len', type=int, default=1024, help='max length of tokenized document (default: 1024)')
    parser.add_argument('--max_target_len', type=int, default=512, help='max length of tokenized summary (default: 512)')

    # -- Seed
    parser.add_argument('--seed', type=int, default=42, help='random seed (default: 42)')

    # -- Wandb
    parser.add_argument('--dotenv_path', default='./wandb.env', help='input your dotenv path')

    args = parser.parse_args()

    seed_everything(args.seed)   
    main(args)

