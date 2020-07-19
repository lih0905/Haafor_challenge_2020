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
        embedded = self.albert(*input)[1]
        output = sigmoid(self.out(embedded))
        return output

class NSP_pool(nn.Module):
    def __init__(self, albert):
        super().__init__()
        self.albert = albert
        embedding_dim = albert.config.to_dict()['hidden_size']
        self.out = nn.Linear(embedding_dim, 1)
        
    def forward(self, input):
        # input = (input_ids, attention_mask, token_type_ids)
        embedded = self.albert(*input)[0]
        # embedded = (batch_size, sentence_length, hidden_dim)
        pool = torch.mean(embedded, dim=1)
        # pool = (batch_size, hidden_dim)

        output = sigmoid(self.out(pool))
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