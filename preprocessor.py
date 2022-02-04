
import re
import math
import numpy as np

class Filtering :
    def __init__(self, min_sen_size) :
        self.sen_size = min_sen_size
        
    def __call__(self, examples) :
        text_data = examples['text']
        sen_list = text_data.split('\n')

        if len(sen_list) < self.sen_size :
            return False
        return True


class Masking :
    def __init__(self, gcp_ratio) :
        self.gcp_ratio = gcp_ratio
        
    def __call__(self, examples) :
        text_data = examples['text']

        source_data = []
        target_data = []
        for text in text_data : 
            sen_list = text.split('\n')

            sen_size = len(sen_list)
            gap_sen_size = math.ceil(self.gcp_ratio * sen_size)
            gap_indices = np.random.choice(sen_size, gap_sen_size, replace=False)
            
            source_text = []
            target_text = []
            for i in range(sen_size) :
                if i in gap_indices :
                    target_text.append(sen_list[i])
                    source_text.append('<mask2>')
                else :
                    source_text.append(sen_list[i])

            source_text = ' '.join(source_text)
            source_text = re.sub(' <mask2>', '<mask2>', source_text)
            source_data.append(source_text)

            target_text = ' '.join(target_text)
            target_data.append(target_text)

        examples['document'] = source_data
        examples['summary'] = target_data
        return examples

class Preprocessor :
    def __init__(self, tokenizer, max_input_length, max_target_length) :
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length

    def __call__(self, examples):
        prefix = self.tokenizer.bos_token + ' '
        inputs = [prefix + doc for doc in examples['document']]

        model_inputs = self.tokenizer(inputs, max_length=self.max_input_length, return_token_type_ids=False, truncation=True)

        # Setup the tokenizer for targets
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(examples["summary"], max_length=self.max_target_length, return_token_type_ids=False, truncation=True)

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs