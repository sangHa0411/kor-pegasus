
import os 
import json
import random
import tensorflow as tf
import pandas as pd
from datasets import Dataset
from tqdm import tqdm

class DataLoader :
    def __init__(self, seed, shard_size) :
        self.seed = seed
        self.shard_size = shard_size

    def load(self, dir_path) :
        documents = []
        summaries = []

        files = os.listdir(dir_path)
        files = [f for f in files if f.endswith(".json")][:10]

        for f in tqdm(files) :
            f_path = os.path.join(dir_path, f)
            with open(f_path, "r") as f :
                dset = json.load(f)

            documents.extend([d["document"] for d in dset])
            summaries.extend([d["summary"] for d in dset])

        df = pd.DataFrame({"document": documents, "summary": summaries})
        dataset = Dataset.from_pandas(df).shuffle(self.seed)
        dataset = [dataset.shard(num_shards=self.shard_size, index=i) for i in range(self.shard_size)]
        return dataset