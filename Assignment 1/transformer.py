import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, dataloader
from transformers_utils import TransformerDataset, TransformerModel, train_transformer, test_transformer, perplexity

dataset = TransformerDataset('Auguste_Maquet.txt', 'glove.6B.100d.txt')
train_sentences, val_sentences, test_sentences = dataset.get_sentences()

batch_size = 32

# hyperparameters
input_dim = 100
vocab_size = len(dataset.word_to_idx)
context_len = dataset.max_len
n_heads = 4
ff_dim = 300
n_layers = 1
dropout = 0.1

num_epochs = 5
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("Device: ", device)

model = TransformerModel(vocab_size, input_dim, context_len, n_heads, ff_dim, n_layers, dropout)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss().to(device)

print("training started")

trained_model = train_transformer(model, train_sentences, val_sentences, batch_size, dataset, num_epochs, criterion, optimizer, device)
torch.save(trained_model,'Transformer.pt')

test_loss = test_transformer(trained_model, test_sentences, batch_size, dataset, criterion, device)

perplexity(trained_model, criterion, test_sentences, dataset, "2021101051-LM3-test-perplexity.txt", device)
perplexity(trained_model, criterion, val_sentences, dataset, "2021101051-LM3-val-perplexity.txt", device)
perplexity(trained_model, criterion, train_sentences, dataset, "2021101051-LM3-train-perplexity.txt", device)
