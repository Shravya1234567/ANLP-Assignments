import torch
import torch.nn as nn
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import pandas as pd
from torch.utils.data import DataLoader
import time
import matplotlib.pyplot as plt
from FT_utils import FtDataset as loradataset, train
from lora_utils import load_lora_model  

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token

model = GPT2LMHeadModel.from_pretrained('gpt2')

lora_r = 128
lora_alpha = 32
lora_dropout = 0.1

lora_model = load_lora_model(model, lora_r, lora_alpha, lora_dropout)
lora_model.print_trainable_parameters()

df_train = pd.read_csv('data/train.csv')
df_val = pd.read_csv('data/validation.csv')

seed = 42
df_train = df_train.sample(n=21000, random_state=seed)
df_val = df_val.sample(n=6000, random_state=seed)

print(f"Train samples: {len(df_train)}")
print(f"Val samples: {len(df_val)}")

max_length = 512
train_dataset = loradataset(df_train, tokenizer, max_length)
val_dataset = loradataset(df_val, tokenizer, max_length)

batch_size = 32

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
lora_model.to(device)

lr = 1e-4
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, lora_model.parameters()), lr=lr)

n_epochs = 10

start_time = time.time()
trained_model, train_loss, val_loss = train(lora_model, train_dataloader, val_dataloader, optimizer, n_epochs, device, tokenizer)
end_time = time.time()

plt.plot(train_loss, label='train loss')
plt.plot(val_loss, label='val loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('lora_loss.png')

print(f"Training time: {end_time - start_time} seconds")

torch.save(trained_model, 'lora_model.pth')