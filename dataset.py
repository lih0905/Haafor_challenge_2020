# https://github.com/barissayil/SentimentAnalysis/blob/master/dataset.py 참고 

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AlbertTokenizer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class NSP_Dataset(Dataset):

    def __init__(self, filename, model_name='albert-base-v1', max_length=512): 
        #Store the contents of the file in a pandas dataframe
        self.df = pd.read_csv(filename, header=0)
        #Initialize the tokenizer for the desired transformer model
        self.tokenizer = AlbertTokenizer.from_pretrained(model_name)
        self.max_len = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        #Select the sentence and label at the specified index in the data frame
        X = self.df.loc[index,'X']
        Y = self.df.loc[index,'Y']
        label = self.df.loc[index,'Label']
        #Preprocess the text to be suitable for the transformer
        res = self.tokenizer.encode_plus(X,Y, return_token_type_ids=True, return_attention_mask=True, truncation=True,
                                         padding='max_length', max_length=self.max_len)
#         input_ids, token_type_ids, attention_mask = res.values()
        return torch.tensor(list(res.values())), label