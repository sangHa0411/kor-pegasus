import os 
from transformers import PegasusTokenizerFast
from tokenizers import pre_tokenizers
from tokenizers.implementations.sentencepiece_unigram import SentencePieceUnigramTokenizer
from tokenizers import decoders, Regex, normalizers, pre_tokenizers, processors

"""
Reference 
    1. https://huggingface.co/course/chapter6/8?fw=pt#building-a-unigram-tokenizer-from-scratch
"""

def prepare_tokenizer(files):
    spl_tokens = ["<cls>", "<sep>", "<unk>", "<pad>", "<mask_1>", "<mask_2>", "<s>", "</s>"]
    tokenizer = SentencePieceUnigramTokenizer()
    
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
    tokenizer.train(files, vocab_size=35000, special_tokens = spl_tokens, unk_token="<unk>")
    return tokenizer

def train_tokenizer(files):
    tokenizer = prepare_tokenizer(files)

    eos_token_id = tokenizer.token_to_id("</s>")
    tokenizer.post_processor = processors.TemplateProcessing(
        single="$A:0 </s>:2",
        pair="$A:0 $B:1 </s>:2",
        special_tokens=[
            ("</s>", eos_token_id)
        ],
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

    files = os.listdir("./corpus")
    files = [os.path.join("./corpus", f) for f in files if f.endswith(".txt")]
    print("The number of files : %d" %len(files))

    tokenizer = train_tokenizer(files)
