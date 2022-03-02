

class Encoder :
    def __init__(self, tokenizer, max_input_length, max_target_length) :
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length

    def __call__(self, examples):
        prefix = self.tokenizer.bos_token + ' '
        inputs = [prefix + doc for doc in examples['document']]

        model_inputs = self.tokenizer(inputs, max_length=self.max_input_length, return_token_type_ids=False, truncation=True)

        # Setup the tokenizer for targets
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(examples["summary"], max_length=self.max_target_length, return_token_type_ids=False, truncation=True)

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs