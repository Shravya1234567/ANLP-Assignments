import torch
import torch.nn as nn
from transformers import GPT2Tokenizer
import pandas as pd
from torch.utils.data import DataLoader
from prompt_utils import PromptDataset, test_prompt
from FT_utils import FtDataset, test_ft

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

df_test = pd.read_csv('data/test.csv')
seed = 42
df_test = df_test.sample(n=3000, random_state=seed)
max_length = 512

original_answers = df_test['highlights'].tolist()

def prompt_model_test():
    prompt = '[SUMMARIZE]'
    num_prompts = len(tokenizer.tokenize(prompt))

    test_dataset = PromptDataset(df_test, tokenizer, max_length, num_prompts)
    batch_size = 32
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = torch.load('prompt_tuning_model.pth')
    model.to(device)

    test_prompt(model,test_dataloader,tokenizer,original_answers,device)

def ft_model_test():
    test_dataset = FtDataset(df_test, tokenizer, max_length)
    batch_size = 32
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = torch.load('ft_model.pth')
    model.to(device)

    test_ft(model,test_dataloader,tokenizer,original_answers,device)

def lora_model_test():
    test_dataset = FtDataset(df_test, tokenizer, max_length)
    batch_size = 32
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = torch.load('lora_model.pth')
    model.to(device)

    test_ft(model,test_dataloader,tokenizer,original_answers,device)


if __name__ == '__main__':
    prompt_model_test()
    ft_model_test()
    lora_model_test()