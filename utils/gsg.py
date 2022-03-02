
import re
import math
import numpy as np

class GapSentenceGeneration :
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
