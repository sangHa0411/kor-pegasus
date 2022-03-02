
import os
import sys
import argparse
import pandas as pd
import numpy as np
import random
import torch

from utils.loader import DataLoader
from utils.gsg import GapSentenceGeneration
from utils.preprocessor import Filtering
from utils.encoder import Encoder
from utils.collator import DataCollatorForPegasus

import wandb
from dotenv import load_dotenv

from trainer.trainer import BucketingTrainer
from model.model import PegasusForPretraining
from transformers.trainer_utils import get_last_checkpoint
from transformers import (
    AutoConfig,
    AutoTokenizer,
    Seq2SeqTrainingArguments
)

def train(args):
    # -- Device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # -- Checkpoint
    model_checkpoint = args.PLM
    
    # -- Datasets
    print('\nLoading Article Data')
    api_key = os.getenv('HUGGINGFACE_KEY')
    article_loader = DataLoader(seed=args.seed, num_proc=4)
    datasets = article_loader.load_data(api_key=api_key)
    print(datasets)

    print('\nFiltering Too Long Text Data')
    data_filter = Filtering(args.min_sen_size)
    datasets = datasets.filter(data_filter)
    print(datasets)

    # -- Preprocessing
    print('\nGenerating Gap Sentences Data')
    gsg = GapSentenceGeneration(0.15)
    datasets = datasets.map(gsg, 
        batched=True,
        num_proc=4,
        load_from_cache_file=True,
    )
    print(datasets)

    # -- Tokenizing
    print('\nTokenizing')
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)
    encoder = Encoder(tokenizer, args.max_input_len, args.max_target_len)
    datasets = datasets.map(encoder, 
        batched=True,
        num_proc=4,
        load_from_cache_file=True,
        remove_columns = datasets.column_names
    )
    print(datasets)

    # -- Configuration
    config = AutoConfig.from_pretrained(model_checkpoint)

    # -- Model
    print('\nLoading Model')
    model = PegasusForPretraining.from_pretrained(model_checkpoint, config=config).to(device)
    print('Model Type : {}'.format(type(model)))

    training_args = Seq2SeqTrainingArguments(
        output_dir = args.output_dir,                                           # output directory
        logging_dir = args.logging_dir,                                         # logging directory
        num_train_epochs = args.epochs,                                         # training epochs
        save_strategy = 'steps',                                                # save strategy
        save_steps = 5000,                                                      # save steps
        logging_strategy = 'steps',                                             # logging strategy
        logging_steps = 1000,                                                   # logging steps                                         
        per_device_train_batch_size = args.train_batch_size,                    # train batch size
        warmup_steps=args.warmup_steps,                                         # warmup steps
        weight_decay=args.weight_decay,                                         # weight decay
        learning_rate = args.learning_rate,                                     # learning rate
        gradient_accumulation_steps=args.gradient_accumulation_steps,           # gradients accumulation steps
        fp16=True if args.fp16 == 1 else False,                                 # fp 16 flag
        overwrite_output_dir=True if args.overwrite_output_dir == 1 else False
    )

    # -- Collator
    data_collator = DataCollatorForPegasus(tokenizer=tokenizer, 
        model=model,
        mlm=True, 
        mlm_probability=0.15
    )

    # -- Trainer
    training_args.size_gap = args.bucketting_size_gap
    trainer = BucketingTrainer(
        model,
        training_args,
        train_dataset=datasets,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    if os.path.exists(args.output_dir) :
        if os.path.isdir(args.output_dir) == False :
            raise ValueError(f"This directory name has already been used")

    # -- Training
    print('\nTraining')
    if os.path.isdir(args.output_dir):
        last_checkpoint = None
        if args.overwrite_output_dir == False :
            last_checkpoint = get_last_checkpoint(args.output_dir)
            if last_checkpoint is None and len(os.listdir(args.output_dir)) > 0:
                raise ValueError(
                    f"Output directory ({args.output_dir}) already exists and is not empty. "
                    "Use --overwrite_output_dir to overcome."
                )

        train_result = trainer.train(resume_from_checkpoint=last_checkpoint)
        print("Train result: ", train_result)
    else :
        train_result = trainer.train()

    trainer.save_model()
        

def main(args):
    load_dotenv(dotenv_path=args.dotenv_path)
    WANDB_AUTH_KEY = os.getenv('WANDB_KEY')
    wandb.login(key=WANDB_AUTH_KEY)

    wandb_name = f"pretraining day1"
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
    parser.add_argument('--train_batch_size', type=int, default=8, help='train batch size (default: 8)')
    parser.add_argument('--min_sen_size', type=int, default=3, help='min sentence size (default: 3)')
    parser.add_argument('--bucketting_size_gap', type=int, default=8, help='bucketting_size_gap (default: 8)')
    parser.add_argument('--warmup_steps', type=int, default=20000, help='number of warmup steps for learning rate scheduler (default: 20000)')
    parser.add_argument('--weight_decay', type=float, default=1e-2, help='streng1th of weight decay (default: 1e-2)')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=16, help='gradient_accumulation_steps of training (default: 16)')
    parser.add_argument('--fp16', type=int, default=0, help='using fp16 (default: 0)')
    parser.add_argument('--overwrite_output_dir', type=int, default=0, help='overwriting output directory')

    # -- Data
    parser.add_argument('--max_input_len', type=int, default=1024, help='max length of tokenized document (default: 1024)')
    parser.add_argument('--max_target_len', type=int, default=256, help='max length of tokenized summary (default: 256)')

    # -- Seed
    parser.add_argument('--seed', type=int, default=42, help='random seed (default: 42)')

    # -- Wandb
    parser.add_argument('--dotenv_path', default='./path.env', help='input your dotenv path')

    args = parser.parse_args()

    seed_everything(args.seed)   
    main(args)

