
import numpy as np
import tensorflow as tf
from dataclasses import dataclass
from typing import Optional, Any, Union

from transformers.utils import PaddingStrategy
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

@dataclass
class DataCollatorForSeq2Seq:
    tokenizer: PreTrainedTokenizerBase
    config: Optional[Any] = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors:Optional[str] = "tf"

    def __call__(self, features):
        """
        features
            1. type : list
            2. element type : dict
                1) input_ids : np.ndarray
                2) attention_mask : np.ndarray
                3) labels : np.ndarray
        """

        max_target_length = max(len(f["labels"]) for f in features)

        for f in features :
            f_label = np.where(f["labels"] == self.config.pad_token_id, self.label_pad_token_id, f["labels"])
            shifted_label = np.zeros(max_target_length)

            shifted_label[1:] = f_label[:-1]
            shifted_label[0] = self.config.bos_token_id
            shifted_label = np.where(shifted_label == self.label_pad_token_id, self.config.pad_token_id, shifted_label)

            f["labels"] = f_label
            f["decoder_input_ids"] = shifted_label.astype(np.int32)
            f["decoder_attention_mask"] = np.where(shifted_label==self.config.pad_token_id, 0, 1)

        features = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            return_tensors=self.return_tensors,
        )
        return features