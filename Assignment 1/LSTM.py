import torch
import torch.nn as nn
import torch.optim as optim
from LSTM_utils import LSTMDataset, LSTM, train_lstm, test_lstm, perplexity

dataset = LSTMDataset('Auguste_Maquet.txt', 'glove.6B.100d.txt')
train_sentences, val_sentences, test_sentences = dataset.get_sentences()

batch_size = 32

# hyperparameters
input_dim = 100
hidden_dim = 300
output_dim = len(dataset.word_to_idx)

num_epochs = 10

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("Device: ", device)

model = LSTM(input_dim, hidden_dim,1, output_dim, 0.1)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss().to(device)

print("training started")

# training
trained_model = train_lstm(model, train_sentences, val_sentences, num_epochs, criterion, optimizer, dataset, batch_size, device)

# saving
torch.save(trained_model, 'LSTM.pt')

# testing
test_loss = test_lstm(trained_model, test_sentences, criterion, dataset, batch_size, device)

# perplexity
perplexity(trained_model, criterion, dataset.test_sentences, dataset, "2021101051-LM2-test-perplexity.txt", device)
perplexity(trained_model, criterion, dataset.val_sentences, dataset, "2021101051-LM2-val-perplexity.txt", device)
perplexity(trained_model, criterion, dataset.train_sentences, dataset, "2021101051-LM2-train-perplexity.txt", device)