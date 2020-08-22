
import time

import torch
from tqdm import tqdm

# calculate the elapsed time
def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

# calculate accuracy
def binary_accuracy(preds, y):
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds==y).float()
    acc = correct.sum() / len(correct)
    return acc

# train, evaluate functions
def train(model, iterator, optimizer, criterion, device):
    epoch_loss = 0
    epoch_acc = 0
    
    model.train()
    
    for batch in tqdm(iterator):
        # batch[0] = (batch_size, 3, hidden_dim)
        # batch[1] = label
        optimizer.zero_grad()
        input = batch[0].permute(1,0,2)
        label = batch[1].type(torch.DoubleTensor).to(device)
        
        input_ids = input[0].to(device)
        attention_mask = input[2].to(device)
        token_type_ids = input[1].to(device)
        
        data = (input_ids, attention_mask, token_type_ids)
        
        predictions = model(data).squeeze(1) # output_dim = 1
        loss = criterion(predictions, label)
        acc = binary_accuracy(predictions, label)
        
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def evaluate(model, iterator, criterion, device):
    epoch_loss = 0
    epoch_acc = 0
    
    model.eval()
    
    with torch.no_grad():
        for batch in tqdm(iterator):
            input = batch[0].permute(1,0,2)
            label = batch[1].type(torch.DoubleTensor).to(device)

            input_ids = input[0].to(device)
            attention_mask = input[2].to(device)
            token_type_ids = input[1].to(device)

            data = (input_ids, attention_mask, token_type_ids)

            predictions = model(data).squeeze(1) 
            loss = criterion(predictions, label)
            acc = binary_accuracy(predictions, label)

            epoch_loss += loss.item()
            epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)
