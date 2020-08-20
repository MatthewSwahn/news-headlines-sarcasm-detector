from transformers import BertTokenizer, BertModel
from torchtext.data import Field, Dataset, Example
import pandas as pd


class BERTHelper():
    def __init__(self, bert_str):
        self.bert_model = BertModel.from_pretrained(bert_str)
        self.bert_tokenizer = BertTokenizer.from_pretrained(bert_str)
        self.max_tokens = self.bert_tokenizer.max_model_input_sizes[bert_str]

    def tokenize_and_cut(self, sentence):
        tokens = self.bert_tokenizer.tokenize(sentence)
        tokens = tokens[:self.max_tokens - 2]
        return tokens

# pandas df to torchtext
# https://stackoverflow.com/questions/52602071/dataframe-as-datasource-in-torchtext
class DataFrameDataset(Dataset):
    """Class for using pandas DataFrames as a datasource"""
    def __init__(self, examples, fields, filter_pred=None):
        """
        Create a dataset from a pandas dataframe of examples and Fields
        Arguments:
            examples pd.DataFrame: DataFrame of examples
            fields {str: Field}: The Fields to use in this tuple. The
                string is a field name, and the Field is the associated field.
            filter_pred (callable or None): use only exanples for which
                filter_pred(example) is true, or use all examples if None.
                Default is None
           """

        self.examples = examples.apply(SeriesExample.fromSeries, args=(fields,), axis=1).tolist()
        if filter_pred is not None:
            self.examples = filter(filter_pred, self.examples)
        self.fields = dict(fields)
        # Unpack field tuples
        for n, f in list(self.fields.items()):
            if isinstance(n, tuple):
                self.fields.update(zip(n, f))
                del self.fields[n]


class SeriesExample(Example):
    """Class to convert a pandas Series to an Example
    helper class for DataFrameDataset"""

    @classmethod
    def fromSeries(cls, data, fields):
        return cls.fromdict(data.to_dict(), fields)

    @classmethod
    def fromdict(cls, data, fields):
        ex = cls()

        for key, field in fields.items():
            if key not in data:
                raise ValueError("Specified key {} was not found in "
                                 "the input data".format(key))
            if field is not None:
                setattr(ex, key, field.preprocess(data[key]))
            else:
                setattr(ex, key, data[key])
        return ex