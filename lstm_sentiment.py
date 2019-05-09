# External libraries
import torch.nn as nn


# Class for the model.
class LSTMSentiment(nn.Module):
    def __init__(self, vocab_size, output_size, embedding_dim, hidden_dim, layers, drop_prob):
        super().__init__()

        self.output_size = output_size
        self.layers = layers
        self.hidden_dim = hidden_dim

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, layers, dropout=drop_prob, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.linear = nn.Linear(hidden_dim, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, hidden):
        batch_size = x.size(0)
        embeds = self.embedding(x)
        lstm_out, hidden = self.lstm(embeds, hidden)
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)
        out = self.dropout(lstm_out)
        sig_out = self.sigmoid(out)
        sig_out = sig_out.view(batch_size, -1)
        sig_out = sig_out[:, -1]

        return sig_out, hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = (weight.new(self.layers, batch_size, self.hidden_dim).zero_(),
                  weight.new(self.layers, batch_size, self.hidden_dim).zero_())

        return hidden
