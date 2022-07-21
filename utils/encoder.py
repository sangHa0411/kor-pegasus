
class Encoder :
    def __init__(self, tokenizer, max_input_length, max_target_length) :
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length

    def __call__(self, examples):
 
        model_inputs = self.tokenizer(examples["document"], 
            padding="max_length",
            max_length=self.max_input_length, 
            return_token_type_ids=False, 
            truncation=True
        )

        with self.tokenizer.as_target_tokenizer() :
            labels = self.tokenizer(examples["summary"],
                max_length=self.max_target_length, 
                return_token_type_ids=False, 
                truncation=True
            )

        decoder_input_ids = []
        decoder_attention_masks = []

        for i, l in enumerate(labels["input_ids"]) :
            remainder = [self.tokenizer.pad_token_id] * (self.max_target_length - len(l))
            labels["input_ids"][i] = l + remainder

            decoder_input_id = [self.tokenizer.bos_token_id] + l[0:-1]
            decoder_attention_mask = [1] * len(decoder_input_id)

            decoder_input_id = decoder_input_id + [self.tokenizer.pad_token_id] * (self.max_target_length - len(decoder_input_id))
            decoder_attention_mask = decoder_attention_mask + [0] * (self.max_target_length - len(decoder_attention_mask))

            decoder_input_ids.append(decoder_input_id)
            decoder_attention_masks.append(decoder_attention_mask) 

        # decoder input
        model_inputs["decoder_input_ids"] = decoder_input_ids
        model_inputs["decoder_attention_mask"] = decoder_attention_masks

        # labels
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs