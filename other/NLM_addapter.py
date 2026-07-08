

class NLM_adapter:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def forward(self, input_text):
        # Tokenize the input text
        inputs = self.tokenizer(input_text, return_tensors='pt')
        
        # Pass the tokenized input through the model
        outputs = self.model(**inputs)
        
        return outputs