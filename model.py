import torch
import torch.nn as nn
from torch import sigmoid
from transformers import AlbertTokenizer, AlbertModel

class NSP(nn.Module):
    def __init__(self, albert):
        super().__init__()
        self.albert = albert
        embedding_dim = albert.config.to_dict()['hidden_size']
        self.out = nn.Linear(embedding_dim, 1)
        
    def forward(self, input):
        # input = (input_ids, attention_mask, token_type_ids)
        with torch.no_grad():
            embedded = self.albert(*input)[1]
        output = self.out(embedded)
        return output

class NSP_pool(nn.Module):
    def __init__(self, albert):
        super().__init__()
        self.albert = albert
        embedding_dim = albert.config.to_dict()['hidden_size']
        self.out = nn.Linear(embedding_dim, 1)
        
    def forward(self, input):
        # input = (input_ids, attention_mask, token_type_ids)
        with torch.no_grad():
            embedded = self.albert(*input)[0]
        # embedded = (batch_size, sentence_length, hidden_dim)
        pool = torch.mean(embedded, dim=1)
        # pool = (batch_size, hidden_dim)

        output = self.out(pool)
        # output = (batch_size, 1)
        return output

class NSP_lin(nn.Module):
    def __init__(self, albert):
        super().__init__()
        self.albert = albert
        embedding_dim = albert.config.to_dict()['hidden_size']
        self.out1 = nn.Linear(embedding_dim, 1)
        self.out2 = nn.Linear(512,1)
        
    def forward(self, input):
        # input = (input_ids, attention_mask, token_type_ids)
        with torch.no_grad():
            embedded = self.albert(*input)[0]
        # embedded = (batch_size, sentence_length, hidden_dim)
        out1 = torch.tanh(self.out1(embedded).squeeze())
        # pool = (batch_size, hidden_dim)
        out2 = self.out2(out1)
        # output = (batch_size, 1)
        return out2

class NSP_gru(nn.Module):
    def __init__(self, albert):
        super().__init__()
        self.albert = albert
        embedding_dim = albert.config.to_dict()['hidden_size']
        hidden_dim = albert.config.to_dict()['hidden_size']
        self.rnn = nn.GRU(embedding_dim, hidden_dim,
                    num_layers = 2,
                    bidirectional = True,
                    batch_first = True,
                    dropout = 0.1)
        self.out = nn.Linear(2*hidden_dim, 1)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, input):
        # input = (input_ids, attention_mask, token_type_ids)
        with torch.no_grad():
            embedded = self.albert(*input)[0]
        # embedded = (batch_size, sentence_length, embedding_dim)

        _, hidden = self.rnn(embedded)
        # hidden = (num_layers * 2, batch_size, hidden_dim)

        hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1))
        # hidden = (batch_size, 2*hidden_dim)


        out = self.out(hidden)
        return out
