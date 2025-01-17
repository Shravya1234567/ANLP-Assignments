import torch
import torch.nn as nn
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import pandas as pd
from torch.utils.data import DataLoader
from prompt_utils import PromptDataset, SoftPromptEmbedding, PromptTuning, train
import time
import matplotlib.pyplot as plt

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token

model = GPT2LMHeadModel.from_pretrained('gpt2')

for param in model.parameters():
    param.requires_grad = False

df_train = pd.read_csv('data/train.csv')
df_val = pd.read_csv('data/validation.csv')

seed = 42
df_train = df_train.sample(n=21000, random_state=seed)
df_val = df_val.sample(n=6000, random_state=seed)

print(f"Train samples: {len(df_train)}")
print(f"Val samples: {len(df_val)}")

max_length = 512

prompt = '[SUMMARIZE]'
num_prompts = len(tokenizer.tokenize(prompt))
print(f"Initial Prompt: {prompt}")
print(f"Number of tokens in prompt: {num_prompts}")
    
train_dataset = PromptDataset(df_train, tokenizer, max_length, num_prompts)
val_dataset = PromptDataset(df_val, tokenizer, max_length, num_prompts)

batch_size = 32
embedding_size = 768

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

soft_prompt_embedding = SoftPromptEmbedding(model, num_prompts, embedding_size, tokenizer, prompt)
prompt_tuning_model = PromptTuning(model, soft_prompt_embedding)

num_params = sum(p.numel() for p in prompt_tuning_model.parameters() if p.requires_grad)
print(f"Number of trainable parameters: {num_params}")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
prompt_tuning_model.to(device)

optimizer = torch.optim.Adam(prompt_tuning_model.soft_prompt_embedding.parameters(), lr=1e-4)
n_epochs = 10

start_time = time.time()
prompt_tuning_model, train_loss, val_loss = train(prompt_tuning_model, train_dataloader, val_dataloader, optimizer, n_epochs, device, tokenizer)
end_time = time.time()

plt.plot(train_loss, label='train loss')
plt.plot(val_loss, label='val loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('prompt_tuning_loss.png')

print(f"Training time: {end_time - start_time} seconds")

torch.save(prompt_tuning_model, 'prompt_tuning_model.pth')