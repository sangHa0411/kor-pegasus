
from tqdm import tqdm
from datasets import load_dataset, concatenate_datasets

class DataLoader :
    def __init__(self, seed) :
        self.seed = seed

    def load_data(self, api_key) :
        data_list = []

        for i in range(10) :
            dataset_name = 'sh110495/kor_newspaper' + str(i+1)
            dataset = load_dataset(dataset_name, use_auth_token=api_key)
            dataset = dataset.remove_columns(['id', 'date', 'topic'])
            data_list.append(dataset['train'])

        for i in range(6) :
            dataset_name = 'sh110495/kor-namuwiki' + str(i+1)
            dataset = load_dataset(dataset_name, use_auth_token=api_key)
            dataset = dataset.remove_columns(['title'])
            data_list.append(dataset['train'])

        for i in range(6) :
            dataset_name = 'sh110495/kor-written-language' + str(i+1)
            dataset = load_dataset(dataset_name, use_auth_token=api_key)
            dataset = dataset.remove_columns(['id', 'date'])
            data_list.append(dataset['train'])

        dataset = load_dataset('sh110495/kor-wikipedia', use_auth_token=api_key)
        dataset = dataset.remove_columns(['title'])
        data_list.append(dataset['train'])

        dataset = load_dataset('sh110495/kor-domain-language', use_auth_token=api_key)
        dataset = dataset.remove_columns(['id', 'type', 'date', 'title'])
        data_list.append(dataset['train'])

        dataset = load_dataset('sh110495/kor-spoken-language', use_auth_token=api_key)
        dataset = dataset.remove_columns(['title', 'publisher', 'date'])
        data_list.append(dataset['train'])

        total_data = concatenate_datasets(data_list)
        total_data = total_data.shuffle(self.seed)
        return total_data
