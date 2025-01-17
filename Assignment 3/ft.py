import torch
import torch.nn as nn
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import pandas as pd
from torch.utils.data import DataLoader
import time
import matplotlib.pyplot as plt
from FT_utils import FtDataset, train

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token

model = GPT2LMHeadModel.from_pretrained('gpt2')

for param in model.parameters():
    param.requires_grad = False

for param in model.lm_head.parameters():
    param.requires_grad = True

for layer in model.transformer.h[-1:]:
    for param in layer.parameters():
        param.requires_grad = True

df_train = pd.read_csv('data/train.csv')
df_val = pd.read_csv('data/validation.csv')

seed = 42
df_train = df_train.sample(n=21000, random_state=seed)
df_val = df_val.sample(n=6000, random_state=seed)

print(f"Train samples: {len(df_train)}")
print(f"Val samples: {len(df_val)}")

max_length = 512
train_dataset = FtDataset(df_train, tokenizer, max_length)
val_dataset = FtDataset(df_val, tokenizer, max_length)

batch_size = 32

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Number of trainable parameters: {num_params}")

optimizer = torch.optim.Adam([
    {'params': model.lm_head.parameters()},
    {'params': model.transformer.h[-1:].parameters()}
], lr=1e-4)

n_epochs = 10

start_time = time.time()
trained_model, train_loss, val_loss = train(model, train_dataloader, val_dataloader, optimizer, n_epochs, device, tokenizer)
end_time = time.time()

plt.plot(train_loss, label='train loss')
plt.plot(val_loss, label='val loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('ft_loss.png')

print(f"Training time: {end_time - start_time} seconds")

torch.save(trained_model, 'ft_model.pth')