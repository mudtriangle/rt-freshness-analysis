# External libraries
import torch
import pandas as pd

# Internal libraries
import lstm_sentiment as lstm
import data_utils as du
import string_processing as sp

# Values necessary to load the network.
vocab = pd.read_csv("vocabulary.txt", names=['ind', 'word'], encoding='iso-8859-1')
vocab = pd.Series(vocab['ind'].values, index=vocab['word']).to_dict()
vocab_size = du.get_vocab_size("vocabulary.txt")

# Load the network.
network = lstm.LSTMSentiment(vocab_size)
network.load_state_dict(torch.load('model'))
network.eval()

# Get user input.
user_sentence = input("Enter a review: ")

# Process user input and convert it to tokens.
user_sentence = sp.normalize(user_sentence)
user_sentence = sp.tokenize(user_sentence)
user_sentence = sp.get_numbers(user_sentence, vocab)
user_sentence = sp.padding(user_sentence, 30)

# Predict and output results.
output, h = network(torch.LongTensor([user_sentence]), network.init_hidden(1))
pred = torch.round(output.squeeze())
if pred.item() == 1:
    print("Fresh")
else:
    print("Rotten")
