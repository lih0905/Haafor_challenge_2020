import torch
import torch.nn as nn
import torch.nn.functional as F
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
                    num_layers = 4,
                    bidirectional = True,
                    batch_first = True,
                    dropout = 0.3)
        self.out = nn.Linear(2*hidden_dim, 1)
        self.dropout = nn.Dropout(0.3)
        
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

class NSP_gru_NoneFreeze(nn.Module):
    def __init__(self, albert):
        super().__init__()
        self.albert = albert
        embedding_dim = albert.config.to_dict()['hidden_size']
        hidden_dim = albert.config.to_dict()['hidden_size']
        self.rnn = nn.GRU(embedding_dim, hidden_dim,
                    num_layers = 4,
                    bidirectional = True,
                    batch_first = True,
                    dropout = 0.3)
        self.out = nn.Linear(2*hidden_dim, 1)
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, input):
        # input = (input_ids, attention_mask, token_type_ids)
        embedded = self.albert(*input)[0]
        # embedded = (batch_size, sentence_length, embedding_dim)

        _, hidden = self.rnn(embedded)
        # hidden = (num_layers * 2, batch_size, hidden_dim)

        hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1))
        # hidden = (batch_size, 2*hidden_dim)


        out = self.out(hidden)
        return out


class CNN(nn.Module):
    def __init__(self, albert, n_filters, filter_sizes, dropout):
        super().__init__()
        self.albert = albert
        embedding_dim = albert.config.to_dict()['hidden_size']
        
        self.convs = nn.ModuleList([nn.Conv2d(in_channels=1,
                                             out_channels=n_filters,
                                             kernel_size=(fs, embedding_dim))
                                   for fs in filter_sizes])
        self.fc = nn.Linear(len(filter_sizes) * n_filters, 1)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input):
        # input = (input_ids, attention_mask, token_type_ids)
        with torch.no_grad():
            embedded = self.albert(*input)[0]
        # embedded = (batch_size, sentence_length, embedding_dim)
        
        embedded = embedded.unsqueeze(1)
        #print_shape('embedded', embedded)
        # embedded = [batch_size, 1, sent_len, emb_dim]
        
        #print_shape('self.convs[0](embedded)', self.convs[0](embedded))
        # self.convs[0](embedded) = [batch_size, n_filters, sent_len-filter_sizes[n]+1, 1 ]
        conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]
        
        #print_shape('F.max_pool1d(conved[0], conved[0].shape[2])', F.max_pool1d(conved[0], conved[0].shape[2]))
        # F.max_pool1d(conved[0], conved[0].shape[2]) = [batch_size, n_filters, 1]
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        
        cat = self.dropout(torch.cat(pooled, dim=1))
        #print_shape('cat', cat)
        # cat = [batch_size, n_filters * len(filter_size)]
        
        res = self.fc(cat)
        #print_shape('res', res)
        # res = [batch_size, output_dim]
        
        return self.fc(cat)