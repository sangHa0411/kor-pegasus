

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
        )

        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(examples["summary"],
                padding="max_length",
                max_length=self.max_target_length, 
                return_token_type_ids=False, 
            )

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs