
import numpy as np
import tensorflow as tf
from dataclasses import dataclass
from typing import Optional, Any, Union

from transformers.utils import PaddingStrategy
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

def shift_tokens_right(input_ids: tf.Tensor, pad_token_id: int, decoder_start_token_id: int):
    shifted_input_ids = tf.zeros(input_ids.shape)

    shifted_input_ids[:, 1:] = tf.identity(input_ids[:, :-1])
    shifted_input_ids[:, 0] = decoder_start_token_id

    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)
    return shifted_input_ids

@dataclass
class DataCollatorForSeq2Seq:

    tokenizer: PreTrainedTokenizerBase
    model: Optional[Any] = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors = "tf"

    def __call__(self, features):
        breakpoint()

        features = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            return_tensors=self.return_tensors,
        )
        return features

    def prepare_decoder_input_ids_from_labels(self, labels: tf.tensor):
        return shift_tokens_right(labels, self.config.pad_token_id, self.config.decoder_start_token_id)