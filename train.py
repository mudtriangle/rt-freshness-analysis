# External libraries
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime as dt

# Internal libraries
import lstm_sentiment as lstm
import data_utils as du

# Constants
TRAIN_SPLIT = 0.8
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
EPOCHS = 10

# Load the preprocessed data.
reviews_df = pd.read_csv("reviews.csv", names=['freshness', 'review'], encoding='iso-8859-1')
reviews_df['review'] = reviews_df['review'].apply(lambda x: x.split(' '))
reviews_df['review'] = reviews_df['review'].apply(lambda x: [int(i) for i in x])

# Make training, test, and validation data loaders.
data = reviews_df['review'].tolist()
labels = reviews_df['freshness'].tolist()
train_loader, test_loader, validate_loader = du.make_loaders(data, labels, BATCH_SIZE, TRAIN_SPLIT)

# Get the size of the vocabulary.
vocab_size = du.get_vocab_size("vocabulary.txt")

# Build network.
network = lstm.LSTMSentiment(vocab_size)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(network.parameters(), lr=LEARNING_RATE)
counter = 0
clip = 5

# Main loop for training.
network.train()
for e in range(EPOCHS):
    h = network.init_hidden(BATCH_SIZE)
    for inputs, labels in train_loader:
        counter += 1

        h = tuple([each.data for each in h])

        network.zero_grad()

        inputs = inputs.type(torch.LongTensor)
        output, h = network(inputs, h)

        loss = criterion(output.squeeze(), labels.float())
        loss.backward()

        nn.utils.clip_grad_norm_(network.parameters(), clip)
        optimizer.step()

        # Every 100 iterations, validate.
        if counter % 100 == 0:
            val_h = network.init_hidden(BATCH_SIZE)
            val_losses = []
            network.eval()

            for val_inputs, val_labels in validate_loader:
                val_h = tuple([each.data for each in val_h])

                val_inputs = val_inputs.type(torch.LongTensor)
                val_output, val_h = network(val_inputs, val_h)
                val_loss = criterion(val_output.squeeze(), val_labels.float())

                val_losses.append(val_loss.item())

            network.train()
            print(dt.now().strftime("%Y-%m-%d %H:%M:%S\n"),
                  "Epoch: {}/{}\n".format(e + 1, EPOCHS),
                  "Step: {}\n".format(counter),
                  "Loss: {:.6f}\n".format(loss.item()),
                  "Val Loss: {:.6f}\n".format(np.mean(val_losses)))

# Run against the test data.
test_losses = []
num_correct = 0

test_h = network.init_hidden(BATCH_SIZE)

# Main loop for testing.
network.eval()
for test_inputs, test_labels in test_loader:
    test_h = tuple([each.data for each in h])

    test_inputs = test_inputs.type(torch.LongTensor)
    test_output, test_h = network(test_inputs, test_h)

    test_loss = criterion(test_output.squeeze(), test_labels.float())
    test_losses.append(test_loss.item())

    pred = torch.round(test_output.squeeze())

    correct_tensor = pred.eq(test_labels.float().view_as(pred))
    correct = np.squeeze(correct_tensor.numpy())
    num_correct += np.sum(correct)

print("Test loss: {:.3f}".format(np.mean(test_losses)))
test_acc = num_correct/len(test_loader.dataset)
print("Test accuracy: {:.3f}".format(test_acc))

# Save model.
torch.save(network.state_dict(), 'model')
