import torch
import pandas as pd
import numpy as np

import lstm_sentiment as lstm
import data_utils as du
import string_processing as sp

DROP_PROB = 0.5

output_size = 1
embedding_dim = 400
hidden_dim = 256
num_layers = 2
vocab = pd.read_csv("vocabulary.txt", names=['ind', 'word'], encoding='iso-8859-1')
vocab = pd.Series(vocab['ind'].values, index=vocab['word']).to_dict()
vocab_size = du.get_vocab_size("vocabulary.txt")

network = lstm.LSTMSentiment(vocab_size, output_size, embedding_dim, hidden_dim, num_layers, DROP_PROB)
network.load_state_dict(torch.load('model'))
network.eval()

user_sentence = input("Enter a review: ")
user_sentence = sp.normalize(user_sentence)
user_sentence = sp.tokenize(user_sentence)
user_sentence = sp.get_numbers(user_sentence, vocab)
user_sentence = sp.padding(user_sentence, 30)

output, h = network(torch.LongTensor([user_sentence]), network.init_hidden(1))
pred = torch.round(output.squeeze())
print('Prediction value, pre-rounding: {:.6f}'.format(output.item()))

if pred.item() == 1:
    print("Fresh")
else:
    print("Rotten")
