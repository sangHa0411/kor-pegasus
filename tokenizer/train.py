import os 
import random
from transformers import PegasusTokenizerFast
## importing the tokenizer and subword BPE trainer
from tokenizers import SentencePieceBPETokenizer, pre_tokenizers

## a pretokenizer to segment the text into words
from tokenizers import decoders, Regex, normalizers, pre_tokenizers, processors

"""
Reference 
    1. https://huggingface.co/course/chapter6/8?fw=pt#building-a-unigram-tokenizer-from-scratch
"""

def prepare_tokenizer():
    spl_tokens = ["<cls>", "<sep>", "<unk>", "<pad>", "<mask_1>", "<mask_2>", "<s>", "</s>"]
    tokenizer = SentencePieceBPETokenizer()
    tokenizer.normalizer = normalizers.Sequence(
        [
            normalizers.Replace("``", '"'),
            normalizers.Replace("''", '"'),
            normalizers.NFKD(),
            normalizers.StripAccents(),
            normalizers.Replace(Regex(" {2,}"), " "),
        ]
    )
    tokenizer.pre_tokenizer = pre_tokenizers.Metaspace()

    tokenizer.train(files, vocab_size=35000, min_frequency=3, special_tokens = spl_tokens)
    return tokenizer

def train_tokenizer(files):
    tokenizer = prepare_tokenizer()

    cls_token_id = tokenizer.token_to_id("<cls>")
    sep_token_id = tokenizer.token_to_id("<sep>")

    tokenizer.post_processor = processors.TemplateProcessing(
        single="$A:0 <sep>:0 <cls>:2",
        pair="$A:0 <sep>:0 $B:1 <sep>:1 <cls>:2",
        special_tokens=[("<sep>", sep_token_id), ("<cls>", cls_token_id)],
    )

    tokenizer.decoder = decoders.Metaspace()

    wrapped_tokenizer = PegasusTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>",
        cls_token="<cls>",
        sep_token="<sep>",
        mask_token="<mask_2>",
        padding_side="right",
    )
    wrapped_tokenizer.save_pretrained("./tokenizer")
    return wrapped_tokenizer

if __name__ == "__main__" :

    files = os.listdir("./documents")
    files = [os.path.join("./documents", f) for f in files if f.endswith(".txt")]
    print("The number of files : %d" %len(files))

    tokenizer = train_tokenizer(files)
