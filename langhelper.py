from transformers import BertTokenizer, BertModel


class BERTHelper():
    def __init__(self, bert_str):
        self.bert_model = BertModel.from_pretrained(bert_str)
        self.bert_tokenizer = BertTokenizer.from_pretrained(bert_str)
        self.max_tokens = self.bert_tokenizer.max_model_input_sizes[bert_str]

    def tokenize_and_cut(self, sentence):
        tokens = self.bert_tokenizer.tokenize(sentence)
        tokens = tokens[:self.max_tokens - 2]
        return tokens
