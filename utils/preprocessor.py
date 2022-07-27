
import re
import numpy as np
import pandas as pd
import multiprocessing
from multiprocessing import Pool

class Preprocessor :
    def __init__(self, ) :
        pass

    def __call__(self, examples) :
        size = len(examples["document"])

        for i in range(size) :
            examples["document"][i] = re.sub("\s+", " ", examples["document"][i]).strip()
            examples["summary"][i] = re.sub("\s+", " ", examples["summary"][i]).strip()

        return examples
