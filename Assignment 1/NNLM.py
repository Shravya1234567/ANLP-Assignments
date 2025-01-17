import torch
import torch.nn as nn
import torch.optim as optim
from NNLM_utils import CreateDataset, NNLM, train_NNLM, test_NNLM, perplexity

dataset = CreateDataset('Auguste_Maquet.txt', 'glove.6B.100d.txt')
train_sentences, val_sentences, test_sentences = dataset.get_sentences()

batch_size = 64

# hyperparameters
input_dim = 500
hidden_dim = 300
output_dim = len(dataset.word_to_idx)

num_epochs = 7

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("Device: ", device)

model = NNLM(input_dim, hidden_dim, output_dim)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss().to(device)

print("training started")

# training
trained_model = train_NNLM(model, train_sentences, val_sentences, num_epochs, criterion, optimizer, dataset, batch_size, device)

# saving
torch.save(trained_model, 'NNLM.pt')

# testing
test_loss = test_NNLM(trained_model, test_sentences, criterion, dataset, batch_size, device)

# perplexity
perplexity(trained_model, criterion, dataset.test_sentences, dataset.embeddings, dataset.word_to_idx, "2021101051-LM1-test-perplexity.txt", device)
perplexity(trained_model, criterion, dataset.val_sentences, dataset.embeddings, dataset.word_to_idx, "2021101051-LM1-val-perplexity.txt", device)
perplexity(trained_model, criterion, dataset.train_sentences, dataset.embeddings, dataset.word_to_idx, "2021101051-LM1-train-perplexity.txt", device)