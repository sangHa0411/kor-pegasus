
from datasets import load_dataset, concatenate_datasets
from preprocessor import NamuwikiPreprocessor, NewspaperPreprocessor

class DataLoader :
    def __init__(self, seed, num_proc=4) :
        self.seed = seed
        self.num_proc = num_proc
        self.news_preprocessor = NewspaperPreprocessor()
        self.namu_preprocessor = NamuwikiPreprocessor()

    def load_data(self, api_key) :
        data_list = []

        # newspaper
        print('\nLoad Korean newspaper data')
        for i in range(10) :
            dataset_name = 'sh110495/kor_newspaper' + str(i+1)
            dataset = load_dataset(dataset_name, use_auth_token=api_key)
            dataset = dataset.map(self.news_preprocessor, 
                batched=True,
                num_proc=self.num_proc,
                load_from_cache_file=True,
                remove_columns = dataset.column_names
            )

            dataset = dataset.remove_columns(['id', 'date', 'topic'])
            data_list.append(dataset['train'])

        # namuwiki data
        print('\nLoad Korean namuwiki data')
        for i in range(6) :
            dataset_name = 'sh110495/kor-namuwiki' + str(i+1)
            dataset = load_dataset(dataset_name, use_auth_token=api_key)
            dataset = dataset.map(self.namu_preprocessor, 
                batched=True,
                num_proc=self.num_proc,
                load_from_cache_file=True,
                remove_columns = dataset.column_names
            )
            dataset = dataset.remove_columns(['title'])
            data_list.append(dataset['train'])

        # written language 
        print('\nLoad Korean written language data')
        for i in range(6) :
            dataset_name = 'sh110495/kor-written-language' + str(i+1)
            dataset = load_dataset(dataset_name, use_auth_token=api_key)
            dataset = dataset.remove_columns(['id', 'date'])
            data_list.append(dataset['train'])

        # science, petition and etc
        print('\nLoad Korean domain language')
        dataset = load_dataset('sh110495/kor-domain-language', use_auth_token=api_key)
        dataset = dataset.remove_columns(['id', 'type', 'date', 'title'])
        data_list.append(dataset['train'])

        # dialogue language
        print('\nLoad Korean Spoken language')
        dataset = load_dataset('sh110495/kor-spoken-language', use_auth_token=api_key)
        dataset = dataset.remove_columns(['title', 'publisher', 'date'])
        data_list.append(dataset['train'])

        total_data = concatenate_datasets(data_list)
        total_data = total_data.shuffle(self.seed)
        return total_data