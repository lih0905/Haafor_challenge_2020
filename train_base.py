# load modules
import time 
import random 

import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader 
from transformers import AlbertModel

from dataset import NSP_Dataset
from model import NSP, NSP_pool, NSP_lin
from utils import epoch_time, binary_accuracy, train, evaluate, send_telegram_message

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
N_EPOCHS = 10
model_name='albert-large-v1'

# load the dataset
train_dataset = NSP_Dataset('Data/train.csv', model_name=model_name, max_length=MAX_LENGTH)
dev_dataset = NSP_Dataset('Data/dev.csv', model_name=model_name, max_length=MAX_LENGTH)
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=8)
dev_dataloader = DataLoader(dev_dataset, batch_size=BATCH_SIZE, num_workers=8)

# load the NSP model
albert = AlbertModel.from_pretrained(model_name)
model = NSP(albert)
MODEL = str(model.__class__).split('.')[1].split("'")[0]
# model_path = f'weights/{MODEL}_{model_name}_batch_{BATCH_SIZE}_epoch_{1}.pt'
# model.load_state_dict(torch.load(model_path))
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
try:
    with open(f'weights/{MODEL}_{model_name}_best_loss.pickle','rb') as f:
        last_epoch, best_valid_loss = pickle.load(f)
except:
    last_epoch = 0
    best_valid_loss = float('infinity')

for epoch in range(last_epoch, N_EPOCHS):
    print(f'Train the {epoch:02}th epoch.')
    print(f'The current best loss is {best_valid_loss:.3f}.')
    start_time = time.time()
    
    train_loss, train_acc = train(model, train_dataloader, optimizer, criterion, device)
    valid_loss, valid_acc = evaluate(model, dev_dataloader, criterion, device)
    
    end_time = time.time()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), f'weights/{MODEL}_{model_name}_batch_{BATCH_SIZE}_epoch_{epoch}.pt')        
        with open(f'weights/{MODEL}_{model_name}_best_loss.pickle', 'wb') as f:
            pickle.dump([epoch+last_epoch+1, best_valid_loss], f)

    msg = f'Model : {MODEL}_{model_name}\n' + \
        f'After training {epoch+last_epoch+1} epochs of {N_EPOCHS} epochs, valid loss = {valid_loss:.3f} and valid acc. = {valid_acc*100:.2f}.\n' + \
        f'Current best valid loss. = {best_valid_loss:.3f}'
    send_telegram_message(msg)

    print(f'Epoch: {epoch+last_epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')