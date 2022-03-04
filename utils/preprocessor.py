
import re
from abc import *

class Filtering :
    def __init__(self, min_sen_size) :
        self.sen_size = min_sen_size
        
    def __call__(self, examples) :
        text_data = examples['text']
        sen_list = text_data.split('\n')

        if len(sen_list) < self.sen_size :
            return False
        return True

# eda 이후에 각 데이터 특성에 알맞게 하위 클래스 생성
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
