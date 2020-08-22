# https://github.com/barissayil/SentimentAnalysis/blob/master/dataset.py 참고 

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AlbertTokenizer, AutoTokenizer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class NSP_Dataset(Dataset):

    def __init__(self, filename, model_name='albert-base-v1', max_length=512, mode='train'): 
        #Store the contents of the file in a pandas dataframe
        self.df = pd.read_csv(filename, header=0)
        #Initialize the tokenizer for the desired transformer model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_len = max_length
        self.mode = mode

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        #Select the sentence and label at the specified index in the data frame
        if self.mode == 'train':
            X = self.df.loc[index,'X_BODY']
            Y = self.df.loc[index,'Y_BODY']
            label = self.df.loc[index,'Label']
        else: 
            X = self.df.loc[index,'NEWS1_BODY']
            Y = self.df.loc[index,'NEWS2_BODY']
   
        #Preprocess the text to be suitable for the transformer
        res = self.tokenizer.encode_plus(X,Y, return_token_type_ids=True, return_attention_mask=True, truncation=True,
                                         padding='max_length', max_length=self.max_len)
        #input_ids, token_type_ids, attention_mask = res.values()
        if self.mode == 'train':
            return torch.tensor(list(res.values())), label
        else:
            return torch.tensor(list(res.values()))
