
import os 
import json
import random
import tensorflow as tf
import pandas as pd
from datasets import Dataset
from tqdm import tqdm

class DataLoader :
    def __init__(self, seed) :
        self.seed = seed

    def load(self, dir_path) :
        documents = []
        summaries = []

        files = os.listdir(dir_path)
        files = [f for f in files if f.endswith(".json")]
        
        # For debugging
        files = [files[0]]

        for f in tqdm(files) :
            f_path = os.path.join(dir_path, f)
            with open(f_path, "r") as f :
                dset = json.load(f)

            documents.extend([d["document"] for d in dset])
            summaries.extend([d["summary"] for d in dset])

        df = pd.DataFrame({"document": documents, "summary": summaries})
        dataset = Dataset.from_pandas(df)
        return dataset.shuffle(self.seed)


def get_tf_datasets(datasets, batch_size) :
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