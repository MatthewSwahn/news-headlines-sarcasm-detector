import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def single_eval(model, text, bert_helper, device):
    '''With our trained model, do a single prediction of sarcastic/not sarcastic
    input:
    model: a text model as part of our classifier class
    text: a sentence (as a string) to evaluate
    bert_helper: class that holds our text cleaning functions

    output: a value between 0 and 1, closer to 0 is real and closer to 1 is sarcastic
    '''
    model.eval()
    text_tok = bert_helper.tokenize_and_cut(text)
    text_tok_ids = [bert_helper.bert_tokenizer.cls_token_id] + \
                   bert_helper.bert_tokenizer.convert_tokens_to_ids(text_tok) + \
                   [bert_helper.bert_tokenizer.sep_token_id]
    text_tensor = torch.LongTensor(text_tok_ids).unsqueeze(0).to(device)
    model_out = model(text_tensor)
    return torch.sigmoid(model_out).item()


class BERTLSTMSentimentLast2(nn.Module):
    '''The class which will use for our model.
    This has a few parts, the bert language model, turning our token indices to a series of bert embeddings, a GRU to produce an output based on the token embeddings,
    and a final linear layer to get a single scalar value from the GRU output. There is also a dropout layer to prevent overfitting the training data.
    '''

    def __init__(self,
                 bert_helper,
                 hidden_dim,
                 output_dim,
                 n_layers,
                 dropout):
        super().__init__()

        # store bert model and tokenizer (from huggingface)
        self.bert_model = bert_helper.bert_model
        self.bert_tokenizer = bert_helper.bert_tokenizer

        # store max tokens needed for model
        self.bert_max_tokens = bert_helper.max_tokens
        self.tokenize_and_cut = bert_helper.tokenize_and_cut

        # dimensions of bert embedding
        bert_dim = self.bert_model.config.to_dict()['hidden_size']

        # instantiate LSTM
        self.rnn = nn.LSTM(bert_dim,
                          hidden_dim,
                          num_layers=n_layers,
                          batch_first=True,
                          dropout=0 if n_layers < 2 else dropout)

        self.out = nn.Linear(hidden_dim * 2, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        # text = [batch size, sent len]
        # no_grad so we don't do backprop on the bert embeddings.
        with torch.no_grad():
            embedded = self.bert_model(text)[0]

        # embedded = [batch size, sent len, emb dim]
        # LSTM returns output, (hidden, cell)
        _, hidden = self.rnn(embedded)

        # hidden = [n layers * n directions, batch size, emb dim]
        hidden = self.dropout(torch.cat((hidden[1][-2, :, :], hidden[1][-1, :, :]), dim=1))

        # hidden = [batch size, hid dim]
        output = self.out(hidden)

        # output = [batch size, out dim]
        return output


class BERTGRUSentiment(nn.Module):
    '''The class which will use for our model.
    This has a few parts, the bert language model, turning our token indices to a series of bert embeddings, a GRU to produce an output based on the token embeddings,
    and a final linear layer to get a single scalar value from the GRU output. There is also a dropout layer to prevent overfitting the training data.
    '''

    def __init__(self,
                 bert_helper,
                 hidden_dim,
                 output_dim,
                 n_layers,
                 dropout):
        super().__init__()

        # store bert model and tokenizer (from huggingface)
        self.bert_model = bert_helper.bert_model
        self.bert_tokenizer = bert_helper.bert_tokenizer

        # dimensions of bert embedding
        bert_dim = self.bert_model.config.to_dict()['hidden_size']

        # instantiate GRU
        self.rnn = nn.GRU(bert_dim,
                          hidden_dim,
                          num_layers=n_layers,
                          batch_first=True,
                          dropout=0 if n_layers < 2 else dropout)

        self.out = nn.Linear(hidden_dim, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        # text = [batch size, sent len]
        # no_grad so we don't do backprop on the bert embeddings.
        with torch.no_grad():
            embedded = self.bert_model(text)[0]

        # embedded = [batch size, sent len, emb dim]

        _, hidden = self.rnn(embedded)

        # hidden = [n layers * n directions, batch size, emb dim]
        hidden = self.dropout(hidden[-1, :, :])

        # hidden = [batch size, hid dim]
        output = torch.sigmoid(self.out(hidden))

        # output = [batch size, out dim]
        return output


class BERTGRUSentimentLast2(nn.Module):
    '''The class which will use for our model.
    This has a few parts, the bert language model, turning our token indices to a series of bert embeddings, a GRU to produce an output based on the token embeddings,
    and a final linear layer to get a single scalar value from the GRU output. There is also a dropout layer to prevent overfitting the training data.
    '''

    def __init__(self,
                 bert_helper,
                 hidden_dim,
                 output_dim,
                 n_layers,
                 dropout):
        super().__init__()

        # store bert model and tokenizer (from huggingface)
        self.bert_model = bert_helper.bert_model
        self.bert_tokenizer = bert_helper.bert_tokenizer

        # store max tokens needed for model
        self.bert_max_tokens = bert_helper.max_tokens
        self.tokenize_and_cut = bert_helper.tokenize_and_cut

        # dimensions of bert embedding
        bert_dim = self.bert_model.config.to_dict()['hidden_size']

        # instantiate GRU
        self.rnn = nn.GRU(bert_dim,
                          hidden_dim,
                          num_layers=n_layers,
                          batch_first=True,
                          dropout=0 if n_layers < 2 else dropout)

        self.out = nn.Linear(hidden_dim * 2, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        # text = [batch size, sent len]
        # no_grad so we don't do backprop on the bert embeddings.
        with torch.no_grad():
            embedded = self.bert_model(text)[0]

        # embedded = [batch size, sent len, emb dim]

        _, hidden = self.rnn(embedded)

        # hidden = [n layers * n directions, batch size, emb dim]
        hidden = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))
        # for some reason multiple layer lstm is a tuple vs gru is a tensor

        # hidden = [batch size, hid dim]
        output = self.out(hidden)

        # output = [batch size, out dim]
        return output


class BERTGRUSentimentPerc(nn.Module):
    '''The class which will use for our model.
    This has a few parts, the bert language model, turning our token indices to a series of bert embeddings, a GRU to produce an output based on the token embeddings,
    and a final linear layer to get a single scalar value from the GRU output. There is also a dropout layer to prevent overfitting the training data.
    '''

    def __init__(self,
                 bert_helper,
                 hidden_dim,
                 l1_dim,
                 l2_dim,
                 output_dim,
                 n_layers,
                 dropout):

        super().__init__()

        # store bert model and tokenizer (from huggingface)
        self.bert_model = bert_helper.bert_model
        self.bert_tokenizer = bert_helper.bert_tokenizer

        # dimensions of bert embedding
        bert_dim = self.bert_model.config.to_dict()['hidden_size']

        # instantiate GRU
        self.rnn = nn.GRU(bert_dim,
                          hidden_dim,
                          num_layers=n_layers,
                          batch_first=True,
                          dropout=0 if n_layers < 2 else dropout)
        self.l1 = nn.Linear(hidden_dim, l1_dim)
        self.l2 = nn.Linear(l1_dim, l2_dim)

        self.out = nn.Linear(l2_dim, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        # text = [batch size, sent len]
        # no_grad so we don't do backprop on the bert embeddings.
        with torch.no_grad():
            embedded = self.bert_model(text)[0]

        # embedded = [batch size, sent len, emb dim]

        _, hidden = self.rnn(embedded)

        # hidden = [n layers * n directions, batch size, emb dim]
        hidden = self.dropout(hidden[-1, :, :])

        # hidden to linear
        output = self.dropout(self.l1(hidden))
        output = self.dropout(self.l2(output))

        # hidden = [batch size, hid dim]
        output = torch.sigmoid(self.out(output))

        # output = [batch size, out dim]
        return output


class BERTGRUSentimentLast2Perc(nn.Module):
    '''The class which will use for our model.
    This has a few parts, the bert language model, turning our token indices to a series of bert embeddings, a GRU to produce an output based on the token embeddings,
    and a final linear layer to get a single scalar value from the GRU output. There is also a dropout layer to prevent overfitting the training data.
    '''

    def __init__(self,
                 bert_helper,
                 hidden_dim,
                 l1_dim,
                 l2_dim,
                 output_dim,
                 n_layers,
                 dropout):
        super().__init__()

        # store bert model and tokenizer (from huggingface)
        self.bert_model = bert_helper.bert_model
        self.bert_tokenizer = bert_helper.bert_tokenizer

        # store max tokens needed for model
        self.bert_max_tokens = bert_helper.max_tokens
        self.tokenize_and_cut = bert_helper.tokenize_and_cut

        # dimensions of bert embedding
        bert_dim = self.bert_model.config.to_dict()['hidden_size']

        # instantiate GRU
        self.rnn = nn.GRU(bert_dim,
                          hidden_dim,
                          num_layers=n_layers,
                          batch_first=True,
                          dropout=0 if n_layers < 2 else dropout)
        self.l1 = nn.Linear(hidden_dim * 2, l1_dim)
        self.l2 = nn.Linear(l1_dim, l2_dim)

        self.out = nn.Linear(l2_dim, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        # text = [batch size, sent len]
        # no_grad so we don't do backprop on the bert embeddings.
        with torch.no_grad():
            embedded = self.bert_model(text)[0]

        # embedded = [batch size, sent len, emb dim]

        _, hidden = self.rnn(embedded)

        # hidden = [n layers * n directions, batch size, emb dim]
        hidden = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))

        # hidden to linear
        output = self.dropout(self.l1(hidden))
        output = self.dropout(self.l2(output))

        # hidden = [batch size, hid dim]
        output = self.out(output)

        # output = [batch size, out dim]
        return output
