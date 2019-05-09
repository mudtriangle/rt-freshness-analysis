# External libraries
from torch.utils.data import DataLoader, TensorDataset
import torch
import pandas as pd


# Split the data in training, test, and validation data.
def make_test_train_validate(data, labels, train_split):
    length = len(data)
    cut_val_train = int(train_split * length)
    cut_val_test = int(len(data[cut_val_train:]) * 0.5 + cut_val_train)

    x_train = data[0:cut_val_train]
    y_train = labels[0:cut_val_train]
    x_test = data[cut_val_train:cut_val_test]
    y_test = labels[cut_val_train:cut_val_test]
    x_validate = data[cut_val_test:]
    y_validate = labels[cut_val_test:]

    return x_train, y_train, x_test, y_test, x_validate, y_validate


# Build each data loader for training, testing, and validating.
def make_loaders(data, labels, batch_size, train_split):
    x_train, y_train, x_test, y_test, x_validate, y_validate = make_test_train_validate(data, labels, train_split)

    train_data = TensorDataset(torch.tensor(x_train), torch.tensor(y_train))
    test_data = TensorDataset(torch.tensor(x_test), torch.tensor(y_test))
    validate_data = TensorDataset(torch.tensor(x_validate), torch.tensor(y_validate))
    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size, drop_last=True)
    test_loader = DataLoader(test_data, shuffle=True, batch_size=batch_size, drop_last=True)
    validate_loader = DataLoader(validate_data, shuffle=True, batch_size=batch_size, drop_last=True)

    return train_loader, test_loader, validate_loader


# Get the size of the vocabulary in the file specified.
def get_vocab_size(fname):
    vocab_df = pd.read_csv("vocabulary.txt", names=['ind', 'word'], encoding='iso-8859-1')

    return vocab_df.shape[0] + 1
