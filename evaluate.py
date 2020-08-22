# load modules
import time 
import random 
import os
from tqdm import tqdm

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader 
from transformers import AlbertModel, BertModel, AutoModel

from dataset import NSP_Dataset
from model import NSP, NSP_pool, NSP_lin, NSP_gru, NSP_gru_NoneFreeze, CNN
from utils import epoch_time, binary_accuracy, evaluate, send_telegram_message

# fix random seed
SEED = 1234

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

# gpu 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# hyperparameters
BATCH_SIZE = 128 
MAX_LENGTH = 512
N_EPOCHS = 20
model_name='albert-large-v2'
#model_name= 'roberta-base'

# load the dataset
ev_dataset = NSP_Dataset('Data/evaluation.csv', model_name=model_name, max_length=MAX_LENGTH, mode='evaluate')
ev_dataloader = DataLoader(ev_dataset, batch_size=BATCH_SIZE, num_workers=4)

# load the NSP model
# albert = AlbertModel.from_pretrained(model_name)
bert = AutoModel.from_pretrained(model_name)
model = NSP_gru(bert)

MODEL = str(model.__class__).split('.')[1].split("'")[0]
model_path = f'weights/NSP_gru_albert-large-v2_batch_128_epoch_18.pt'
model.load_state_dict(torch.load(model_path))
model.to(device)

# load the optimizer and loss function
criterion = nn.BCEWithLogitsLoss()
criterion.to(device)


print(f'Evaluate the result.')
start_time = time.time()


result = []
model.eval()

with torch.no_grad():
    for batch in tqdm(ev_dataloader):
        input = batch.permute(1,0,2)

        input_ids = input[0].to(device)
        attention_mask = input[2].to(device)
        token_type_ids = input[1].to(device)

        data = (input_ids, attention_mask, token_type_ids)

        predictions = model(data).squeeze(1) 
        preds = torch.round(torch.sigmoid(predictions))
        result.append(preds)

result = torch.cat(result).cpu().int()
length = len(result)
end_time = time.time()
epoch_mins, epoch_secs = epoch_time(start_time, end_time)


print(f'Evaluate Time: {epoch_mins}m {epoch_secs}s')
print(f'Total length of evaluate data set is {length}.')

result_a = torch.ones(length, dtype=torch.int)
result_a -= result
result = result.numpy()
result_a = result_a.numpy()
indices = np.arange(1,length+1)
d = {'BEF':result_a, 'AFT':result}

df = pd.DataFrame(d, index=indices)
df.to_csv('Data/answer.csv', header=None)

