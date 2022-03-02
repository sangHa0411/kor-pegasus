
import datasets
from sampler import BucketSampler
from transformers import Seq2SeqTrainer
from torch.utils.data import DataLoader
from transformers.file_utils import is_datasets_available

class BucketingTrainer(Seq2SeqTrainer):    
    def get_train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
            train_dataset = self._remove_unused_columns(train_dataset, description="training")

        train_sampler = BucketSampler(dataset=train_dataset, 
            batch_size=self.args.train_batch_size, 
            size_gap=self.args.size_gap
        )

        return DataLoader(
            train_dataset,                                  # dataset
            batch_sampler=train_sampler.sample(),           # sampler
            collate_fn=self.data_collator,                  # collator
            num_workers=self.args.dataloader_num_workers,   # workers
            pin_memory=self.args.dataloader_pin_memory,
        )