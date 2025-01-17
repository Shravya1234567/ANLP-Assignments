import numpy as np
import torch
import torch.nn as nn
from NNLM_utils import CreateDataset
import random

class LSTMDataset(CreateDataset):
    def __init__(self, data_path, embedding_path, cut_off_freq=1, word_to_idx=None, idx_to_word=None):

        super().__init__(data_path, embedding_path, cut_off_freq, word_to_idx, idx_to_word)
        self.word_to_idx['<pad>'] = len(self.word_to_idx)

    def create_data(self, sentences):
        X = []
        y = []

        max_len = max([len(sentence) for sentence in sentences])
        for i in range(len(sentences)):
            sentences[i] = sentences[i] + ['<pad>']*(max_len - len(sentences[i]))

        for sentence in sentences:
            embedding_x = sentence[:-1]
            embedding_x = [self.embeddings[word] for word in embedding_x]
            X.append(embedding_x)

            embedding_y = sentence[1:]
            embedding_y = [self.word_to_idx[word] for word in embedding_y]
            embedding_y_one_hot = np.zeros((len(embedding_y), len(self.word_to_idx)))
            for i in range(len(embedding_y)):
                embedding_y_one_hot[i][embedding_y[i]] = 1
            y.append(embedding_y_one_hot)

        X = np.array(X)
        y = np.array(y)

        X = torch.from_numpy(X).float()
        y = torch.from_numpy(y).float()

        return X, y
    
class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers, output_dim, dropout):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
       
        h_0 = torch.zeros(self.n_layers, x.size(0), self.hidden_dim).to(x.device)
        c_0 = torch.zeros(self.n_layers, x.size(0), self.hidden_dim).to(x.device)

        r_out, (_,_) = self.lstm(x, (h_0, c_0))
        output = self.fc(r_out)
        return output
    
def train_lstm(model, train_sentences, val_sentences, num_epochs, criterion, optimizer, dataobj, batch_size, device):
    model = model.to(device)

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        total_steps = 0
        random.shuffle(train_sentences)
        for i in range(0, len(train_sentences), batch_size):
            sentences = train_sentences[i:i+batch_size]
            optimizer.zero_grad()
            X, y = dataobj.create_data(sentences)
            X = X.to(device)
            y = y.to(device)
            y_pred = model(X)
            y = y.view(-1, y.size(2))
            y_pred = y_pred.view(-1, y_pred.size(2))
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            total_steps += 1
        train_loss /= total_steps

        val_loss = 0
        total_steps = 0
        model.eval()
        with torch.no_grad():
            for i in range(0, len(val_sentences), batch_size):
                sentences = val_sentences[i:i+batch_size]
                X, y = dataobj.create_data(sentences)
                X = X.to(device)
                y = y.to(device)
                y_pred = model(X)
                y = y.view(-1, y.size(2))
                y_pred = y_pred.view(-1, y_pred.size(2))
                loss = criterion(y_pred, y)
                val_loss += loss.item()
                total_steps += 1
            val_loss /= total_steps
        print('Epoch: {}, Training Loss: {}, Validation Loss: {}'.format(epoch+1, train_loss, val_loss))
    
    return model

def test_lstm(model, test_sentences, criterion, dataobj, batch_size, device):
    model = model.to(device)
    test_loss = 0
    total_steps = 0
    model.eval()
    with torch.no_grad():
        for i in range(0, len(test_sentences), batch_size):
            sentences = test_sentences[i:i+batch_size]
            X, y = dataobj.create_data(sentences)
            X = X.to(device)
            y = y.to(device)
            y_pred = model(X)
            y = y.view(-1, y.size(2))
            y_pred = y_pred.view(-1, y_pred.size(2))
            loss = criterion(y_pred, y)
            test_loss += loss.item()
            total_steps += 1
        test_loss /= total_steps
    print('Test Loss: {}'.format(test_loss))
    
    return test_loss

def perplexity(model, criterion, sentences, dataobj, file_path, device):
    avg_perplexity = 0
    model = model.to(device)
    model.eval()
    for sentence in sentences:
        X, y = dataobj.create_data([sentence])
        X = X.to(device)
        y = y.to(device)
        y_pred = model(X)
        y = y.view(-1, y.size(2))
        y_pred = y_pred.view(-1, y_pred.size(2))
        loss = criterion(y_pred, y)
        perplexity = torch.exp(loss)
        avg_perplexity += perplexity.item()
        sentence = sentence[1:-1]
        with open(file_path, 'a') as f:
            f.write(' '.join(sentence) + '\t' + str(perplexity.item()) + '\n')

    avg_perplexity /= len(sentences)
    with open(file_path, 'a') as f:
        f.write('Average Perplexity: ' + str(avg_perplexity) + '\n')