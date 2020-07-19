# load modules
import time 
import random 

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader 
from transformers import AlbertModel

from dataset import NSP_Dataset
from model import NSP, NSP_pool, NSP_lin
from utils import epoch_time, binary_accuracy, train, evaluate

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
N_EPOCHS = 5
model_name='albert-large-v1'

# load the dataset
train_dataset = NSP_Dataset('Data/train.csv', model_name=model_name, max_length=MAX_LENGTH)
dev_dataset = NSP_Dataset('Data/dev.csv', model_name=model_name, max_length=MAX_LENGTH)
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE)
dev_dataloader = DataLoader(dev_dataset, batch_size=BATCH_SIZE)

# load the NSP model
albert = AlbertModel.from_pretrained(model_name)
model = NSP_lin(albert)
model.to(device)

# freeze the parameters of albert
for name, param in model.named_parameters():
    if name.startswith('albert'):
        param.requires_grad = False

# load the optimizer and loss function
optimizer = optim.Adam(model.parameters())
criterion = nn.BCEWithLogitsLoss()
criterion.to(device)

# train
best_valid_loss = float('inf')

for epoch in range(N_EPOCHS):
    start_time = time.time()
    
    train_loss, train_acc = train(model, train_dataloader, optimizer, criterion, device)
    valid_loss, valid_acc = evaluate(model, dev_dataloader, criterion, device)
    
    end_time = time.time()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), f'nsp-model_batch_{BATCH_SIZE}_epoch_{epoch}.pt')
    
    print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')