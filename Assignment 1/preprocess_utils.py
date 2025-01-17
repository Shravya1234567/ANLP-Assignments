import re
import contractions
import nltk
import numpy as np


def preprocess(text):

    text = re.sub(' +', ' ', text)
    text = text.lower()
    preprocessed_text = contractions.fix(text)
    return preprocessed_text

def tokenize(text):
    sentences = nltk.sent_tokenize(text)
    tokens = [nltk.word_tokenize(sentence) for sentence in sentences]
    return tokens

def get_tokens(file_path):

    with open(file_path, 'r') as file:
        lines = file.read()
    
    preprocessed_text = preprocess(lines)
    tokens = tokenize(preprocessed_text)

    for i in range(len(tokens)):
            tokens[i] = [token for token in tokens[i] if any(c.isalnum() for c in token)]

    tokens = [sentence for sentence in tokens if len(sentence) > 5]
    print(len(tokens))
    return tokens

def get_embeddings(file_path):

    embeddings = {}

    with open(file_path, 'r') as f:
         for line in f:
            values = line.split()
            coefs = np.asarray(values[-100:], dtype='float32')
            word = ' '.join(values[:-100])
            embeddings[word] = coefs

    embeddings['<unk>'] = np.ones(100)
    embeddings['<s>'] = np.zeros(100)
    embeddings['</s>'] = np.zeros(100)
    embeddings['<pad>'] = np.zeros(100)

    return embeddings 