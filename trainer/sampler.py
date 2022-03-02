import parmap
import random
import collections
import multiprocessing
from operator import itemgetter

class BucketSampler :
    def __init__(self, dataset, batch_size, size_gap) :
        self.dataset = dataset
        self.batch_size = batch_size
        self.size_gap = size_gap
        self.num_cores = multiprocessing.cpu_count()
     
    def sample(self) :
        batch_map = collections.defaultdict(list)
        idx_list = []
        batch_index = []

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
