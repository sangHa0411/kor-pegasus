
from tqdm import tqdm
from datasets import load_dataset, concatenate_datasets

class DataLoader :
    def __init__(self, data_size, seed) :
        self.data_size = data_size
        self.seed = seed
        self.prefix = 'sh110495/kor_newspaper'

    def load_data(self, api_key) :
        data_list = []
        for i in range(self.data_size) :
            dataset_name = self.prefix + str(i+1)
            dataset = load_dataset(dataset_name, use_auth_token=api_key)
            data = dataset['train']
            data_list.append(data)

        total_data = concatenate_datasets(data_list)
        total_data = total_data.shuffle(self.seed)
        return total_data