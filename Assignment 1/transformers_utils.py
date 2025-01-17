import numpy as np
import torch
import torch.nn as nn
from NNLM_utils import CreateDataset
import math

class TransformerDataset(CreateDataset):
    def __init__(self, data_path, embedding_path, cut_off_freq=1, word_to_idx=None, idx_to_word=None):

        super().__init__(data_path, embedding_path, cut_off_freq, word_to_idx, idx_to_word)
        self.word_to_idx['<pad>'] = len(self.word_to_idx)
        self.max_len = self.get_maxlen()

    def get_maxlen(self):
        max_len = max([len(sentence) for sentence in self.sentences])
        return max_len

    def create_data(self, sentences):
        X = []
        y = []

        for i in range(len(sentences)):
            sentences[i] = sentences[i] + ['<pad>']*(self.max_len - len(sentences[i]))

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
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
    

class TransformerModel(nn.Module):
    def __init__(self, vocab_size, input_dim, contex_len, num_heads, ff_dim, n_layers, dropout):

        super(TransformerModel, self).__init__()

        self.positional_encoder = PositionalEncoding(input_dim, contex_len)
        self.decoder_layer = nn.TransformerDecoderLayer(d_model = input_dim, nhead = num_heads, dim_feedforward = ff_dim, dropout = dropout, batch_first=True)
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers = n_layers)
        self.fc = nn.Linear(input_dim, vocab_size)
        self.input_dim = input_dim

    def generate_square_subsequent_mask(self, sz):
        mask = torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)
        return mask

    def forward(self, x):
        mask = self.generate_square_subsequent_mask(x.size(1))
        x = x * math.sqrt(self.input_dim)
        x = self.positional_encoder(x)
        output = self.decoder(tgt=x, memory=x, tgt_mask=mask)
        output = self.fc(output)
        return output
    
def train_transformer(model, train_sentences, val_sentences, batch_size, dataobj, n_epochs, criterion, optimizer, device):

    model.to(device)

    for epoch in range(n_epochs):
        model.train()
        train_loss = 0
        total_steps = 0
        for i in range(0, len(train_sentences), batch_size):
            sentences = train_sentences[i:i+batch_size]
            X, y = dataobj.create_data(sentences)
            optimizer.zero_grad()
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
        print(f'Epoch {epoch} | Train Loss: {train_loss} | Val Loss: {val_loss}')

    return model
        
def test_transformer(model, test_sentences, batch_size, dataobj, criterion, device):
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
    with torch.no_grad():
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
    
