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