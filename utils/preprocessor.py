
import re
from abc import *
from nltk.tokenize import sent_tokenize

class Filtering :
    def __init__(self, min_sen_size) :
        self.sen_size = min_sen_size
        
    def __call__(self, examples) :
        text_data = examples['text']
        sen_list = text_data.split('\n')

        if len(sen_list) < self.sen_size :
            return False
        return True

class Preprocessor(metaclass=ABCMeta) :
    def __init__(self, ) :
        pass

    def __call__(self, dataset) :
        docs = []
        size = len(dataset['text'])

        for i in range(size) :
            raw_text = dataset['text']
            text = self.preprocess(raw_text)
            docs.append(text)
            
        dataset['text'] = docs
        return dataset

    @abstractmethod
    def preprocess(self, text) :
        return text

class NewspaperPreprocessor(Preprocessor) :

    def preprocess(self, text) :
        sen_list = text.split('\n')

        sen_dataset = []
        for sen in sen_list :
            sens = sent_tokenize(sen)
            sen_dataset.extend(sens)

        text = '\n'.join(sen_dataset)
        return text

class NamuwikiPreprocessor(Preprocessor) :

    def preprocess(self, text) :
        sen_list = text.split('\n')
        sen_list = [sen for sen in sen_list if '= =' not in sen]
        text = '\n'.join(sen_list)
        return text

