
import os 
import json
import pandas as pd
from datasets import Dataset

class DataLoader :
    def __init__(self, seed) :
        self.seed = seed

    def load(self, dir_path) :
        documents = []
        summaries = []

        files = os.listdir(dir_path)
        files = [f for f in files if f.endswith(".json")]

        for f in files :
            f_path = os.path.join(dir_path, f)
            with open(f_path, "r") as f :
                dset = json.load(f)

            documents.extend([d["document"] for d in dset])
            summaries.extend([d["summary"] for d in dset])

        df = pd.DataFrame({"document": documents, "summary": summaries})
        dataset = Dataset.from_pandas(df)
        return dataset.shuffle(self.seed)
