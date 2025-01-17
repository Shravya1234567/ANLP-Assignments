import numpy as np
from preprocess_utils import get_tokens, get_embeddings
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import random

class CreateDataset(Dataset):

    def __init__(self, data_path, embedding_path, cut_off_freq = 1, word_to_idx = None, idx_to_word = None):
        self.sentences = get_tokens(data_path)
        self.embeddings = get_embeddings(embedding_path)
        self.cut_off_freq = cut_off_freq
        self.word_to_idx = word_to_idx
        self.idx_to_word = idx_to_word
        self.train_sentences = None
        self.test_sentences = None
        self.val_sentences = None

        self.add_start_end_tokens()
        self.train_test_val_split()
        self.add_unks()
        if self.word_to_idx is None:
            self.create_vocab()
        self.add_unks_test_val()
        print("preprocessing done")

    def add_start_end_tokens(self):
        for i in range(len(self.sentences)):
            self.sentences[i] = ['<s>'] + self.sentences[i] + ['</s>']
        return self.sentences
    
    def train_test_val_split(self):
        random.shuffle(self.sentences)
        train_size = int(0.7 * len(self.sentences))
        val_size = int(0.1 * len(self.sentences))
        self.train_sentences = self.sentences[:train_size]
        self.val_sentences = self.sentences[train_size:train_size+val_size]
        self.test_sentences = self.sentences[train_size+val_size:]
        return self.train_sentences, self.val_sentences, self.test_sentences
        
    def add_unks(self):
        freq = {}
        for sentence in self.train_sentences:
            for word in sentence:
                if word in freq:
                    freq[word] += 1
                else:
                    freq[word] = 1

        for i in range(len(self.train_sentences)):
            for j in range(len(self.train_sentences[i])):
                if freq[self.train_sentences[i][j]] < self.cut_off_freq or self.train_sentences[i][j] not in self.embeddings:
                    self.train_sentences[i][j] = '<unk>'

        return self.train_sentences
    
    def create_vocab(self):
        self.word_to_idx = {}
        self.idx_to_word = {}
        idx = 0
        for sentence in self.train_sentences:
            for word in sentence:
                if word not in self.word_to_idx:
                    self.word_to_idx[word] = idx
                    self.idx_to_word[idx] = word
                    idx += 1
    
        return self.word_to_idx, self.idx_to_word
    
    def add_unks_test_val(self):
        for i in range(len(self.val_sentences)):
            for j in range(len(self.val_sentences[i])):
                if self.val_sentences[i][j] not in self.word_to_idx:
                    self.val_sentences[i][j] = '<unk>'
        for i in range(len(self.test_sentences)):
            for j in range(len(self.test_sentences[i])):
                if self.test_sentences[i][j] not in self.word_to_idx:
                    self.test_sentences[i][j] = '<unk>'

    def create_data(self, sentences):
        X = []
        y = []
        for sentence in sentences:
            for i in range(5, len(sentence)):
                embeddings_5_gram = [self.embeddings[sentence[j]] for j in range(i-5, i)]
                embeddings_5_gram = np.array(embeddings_5_gram).flatten()
                X.append(embeddings_5_gram)
                y_one = np.zeros(len(self.word_to_idx))
                y_one[self.word_to_idx[sentence[i]]] = 1
                y.append(y_one)
        X = np.array(X)
        y = np.array(y)
        X = torch.tensor(X)
        y = torch.tensor(y)
        return X, y
    
    def get_sentences(self):
        return self.train_sentences, self.val_sentences, self.test_sentences
    

class NNLM(nn.Module):
    def __init__(self,input_dim, hidden_dim, output_dim):
        super(NNLM, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.tanh = nn.Tanh()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x1 = self.fc1(x)
        x2 = self.tanh(x1)
        x3 = self.fc2(x2)
        return x3
    
def train_NNLM(model, train_sentences, val_sentences, num_epochs, criterion, optimizer, dataobj, batch_size, device):

    model = model.to(device)

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        total = 0
        random.shuffle(train_sentences)
        for i in range(0, len(train_sentences), batch_size):
            sentences = train_sentences[i:i+batch_size]
            optimizer.zero_grad()
            X, y = dataobj.create_data(sentences)
            X = X.to(device).float()
            y = y.to(device).float()
            y_pred = model(X)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            total += 1
        train_loss /= total

        val_loss = 0
        total = 0
        model.eval()
        with torch.no_grad():
            for i in range(0, len(val_sentences), batch_size):
                sentences = val_sentences[i:i+batch_size]
                X, y = dataobj.create_data(sentences)
                X = X.to(device).float()
                y = y.to(device).float()
                y_pred = model(X)
                loss = criterion(y_pred, y)
                val_loss += loss.item()
                total += 1
            val_loss /= total
        print('Epoch: {}, Training Loss: {}, Validation Loss: {}'.format(epoch+1, train_loss, val_loss))
    
    return model

def test_NNLM(model, test_sentences, criterion, dataobj, batch_size, device):
    model = model.to(device)
    test_loss = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for i in range(0, len(test_sentences), batch_size):
            sentences = test_sentences[i:i+batch_size]
            X, y = dataobj.create_data(sentences)
            X = X.to(device).float()
            y = y.to(device).float()
            y_pred = model(X)
            loss = criterion(y_pred, y)
            test_loss += loss.item()
            total += 1
        test_loss /= total
    print('Test Loss: {}'.format(test_loss))
    
    return test_loss

def get_perplexity(batch_data, model, criterion, device):
    model = model.to(device)
    perplexity = 0
    model.eval()
    with torch.no_grad():
        X, y = batch_data
        X = X.to(device).float()
        y = y.to(device).float()
        y_pred = model(X)
        loss = criterion(y_pred, y)
        perplexity = torch.exp(loss)
    return perplexity

def perplexity(model, criterion, sentences, embeddings, word_to_idx, file_path, device):
    avg_perplexity = 0
    for sentence in sentences:
        X = []
        y = []
        for i in range(5, len(sentence)):
            embeddings_5_gram = [embeddings[sentence[j]] for j in range(i-5, i)]
            embeddings_5_gram = np.array(embeddings_5_gram).flatten()
            X.append(embeddings_5_gram)
            y_one = np.zeros(len(word_to_idx))
            y_one[word_to_idx[sentence[i]]] = 1
            y.append(y_one)
        X = np.array(X)
        y = np.array(y)
        X = torch.tensor(X)
        y = torch.tensor(y)
        batch_data = (X, y)
        perplexity = get_perplexity(batch_data, model, criterion, device)
        avg_perplexity += perplexity.item()
        sentence = sentence[1:-1]
        with open(file_path, 'a') as f:
            f.write(' '.join(sentence) + '\t' + str(perplexity.item()) + '\n')

    avg_perplexity /= len(sentences)
    with open(file_path, 'a') as f:
        f.write('Average Perplexity: ' + str(avg_perplexity) + '\n')
