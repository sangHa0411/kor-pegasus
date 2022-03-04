
import parmap
import random
import datasets
import collections
import multiprocessing
from operator import itemgetter
from transformers import Seq2SeqTrainer
from torch.utils.data import DataLoader
from transformers.file_utils import is_datasets_available

class BucketSampler :
    def __init__(self, dataset, batch_size, size_gap) :
        self.dataset = dataset
        self.batch_size = batch_size
        self.size_gap = size_gap
        self.num_cores = multiprocessing.cpu_count()
     
    def __call__(self) :
        idx_list = []
        batch_index = []
        batch_map = collections.defaultdict(list)

        data_size = len(self.dataset)
        len_data = parmap.map(self.get_length, self.dataset,  pm_pbar=True, pm_processes=self.num_cores)

        for idx in range(data_size) :
            src_idx, tar_idx = len_data[idx]
            
            src_group = src_idx // self.size_gap
            tar_group = tar_idx // self.size_gap
            batch_map[src_group, tar_group].append(idx)
            
        batch_key = list(batch_map.keys())
        batch_key = sorted(batch_key, key=itemgetter(0,1), reverse=True) 

        # sorting idx list based on size group
        for key in batch_key :
            idx_list.extend(batch_map[key])
    
        # slicing batch_size
        for i in range(0, data_size, self.batch_size) :
            batch_index.append(idx_list[i:i+self.batch_size])
    
        random.shuffle(batch_index)
        return batch_index

    def get_length(self, data) :
        return (len(data['input_ids']), len(data['labels']))

class BucketingTrainer(Seq2SeqTrainer):    
    def get_train_dataloader(self) :
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
            train_dataset = self._remove_unused_columns(train_dataset, description="training")

        breakpoint()
        # bucketing using input sentence length and label sentence length
        train_sampler = BucketSampler(dataset=train_dataset, 
            batch_size=self.args.train_batch_size, 
            size_gap=self.args.size_gap
        )
        breakpoint()

        return DataLoader(
            train_dataset,                                  # dataset
            batch_sampler=train_sampler,                    # sampler
            collate_fn=self.data_collator,                  # collator
            num_workers=self.args.dataloader_num_workers,   # workers
            pin_memory=self.args.dataloader_pin_memory,
        )