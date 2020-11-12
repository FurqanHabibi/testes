# python3.5 buildtagger.py <train_file_absolute_path> <model_file_absolute_path>

import os
import math
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import random
import time
import copy

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pack_sequence, pad_sequence, PackedSequence
from torch.utils.data import Dataset, DataLoader

## Hyperparameters ##
TRAIN_SPLIT = 0.7
BATCH_SIZE = 32
LSTM_LAYERS = 1
WORD_EMBEDDING_SIZE = 64
LSTM_FEATURE_SIZE = 64
CONV_OUT_CHANNEL = 64
CONV_KERNEL_SIZE = 5
DROPOUT = 0.5
EPOCH = 20

max_word_len = 64

tags = ['CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD', 'NN', 'NNP', 'NNPS', 'NNS', 'PDT', 'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'SYM', 'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'WDT', 'WP', 'WP$', 'WRB', '$', '#', '``', "''", '-LRB-', '-RRB-', ',', '.', ':']
tag_to_idx = {tag: idx for idx, tag in enumerate(tags)}

def read_file(filename, with_tag=True, upper=True):
    with open(filename, 'r') as reader:
        lines = reader.readlines()
    
    feats, labels = [], []
    for line in lines:
        words, tags = [], []
        if upper:
            line = line.upper()
        for word_tag in line.strip().split(" "):
            if with_tag:
                word_tag_split = word_tag.split("/")
                word, tag = "/".join(word_tag_split[:-1]), word_tag_split[-1]
                tags.append(tag)
            else:
                word = word_tag
            words.append(word)
        if with_tag:
            labels.append(tags)
        feats.append(words)
    
    if with_tag:
        return feats, labels
    else:
        return feats

def split_feats_labels(feats, labels, split_pct):
    feats_labels = list(zip(feats, labels))
    random.shuffle(feats_labels)
    train_feats_labels, val_feats_labels = feats_labels[:int(len(feats_labels)*split_pct)], feats_labels[int(len(feats_labels)*split_pct):]
    train_feats, train_labels = map(list,zip(*train_feats_labels))
    val_feats, val_labels = map(list,zip(*val_feats_labels))
    return train_feats, train_labels, val_feats, val_labels

def build_word_to_idx(feats):
    word_to_idx = {}
    for words in feats:
        for word in words:
            if word not in word_to_idx:
                word_to_idx[word] = len(word_to_idx)
    return word_to_idx

def indexize(data_list_of_list, data_to_idx):
    def get_idx(data):
        if data not in data_to_idx:
            return len(data_to_idx)
        return data_to_idx[data]

    index_list_of_tensor = []
    for data_list in data_list_of_list:
        index_list = []
        for data in data_list:
            index_list.append(get_idx(data))
        index_tensor = torch.tensor(index_list, dtype=torch.long)
        index_list_of_tensor.append(index_tensor)
    return index_list_of_tensor

def integerize(data_list_of_list):
    integer_list_of_tensor = []
    for data_list in data_list_of_list:
        integer_list_of_list = []
        for data in data_list:
            integer_list = [-1]*max_word_len
            for i, c in enumerate(data):
                integer_list[i] = ord(c)
            integer_list_of_list.append(integer_list)
        integer_list_of_tensor.append(torch.tensor(integer_list_of_list, dtype=torch.float))
    return integer_list_of_tensor

class MyDataset(Dataset):
    
    def __init__(self, features_int, features_idx, labels):
        self.features_int = features_int
        self.features_idx = features_idx
        self.labels = labels

    def __len__(self):
        return len(self.features_int)

    def __getitem__(self, idx):
        return (self.features_int[idx], self.features_idx[idx], self.labels[idx])

def collate(batch_data):
    features_int = []
    features_idx = []
    labels = []
    for feature_int, feature_idx, label in batch_data:
        features_int.append(feature_int)
        features_idx.append(feature_idx)
        labels.append(label)
    features_int.sort(reverse=True, key=lambda t: t.size()[0])
    features_idx.sort(reverse=True, key=lambda t: t.size()[0])
    labels.sort(reverse=True, key=lambda t: t.size()[0])
    return (features_int, features_idx, labels, len(batch_data))

def build_dataset(filename):
    feats, labels = read_file(filename)

    train_feats, train_labels, val_feats, val_labels = split_feats_labels(feats, labels, TRAIN_SPLIT)

    word_to_idx = build_word_to_idx(train_feats)

    train_feats_idx, train_labels, val_feats_idx, val_labels = indexize(train_feats, word_to_idx), indexize(train_labels, tag_to_idx), indexize(val_feats, word_to_idx), indexize(val_labels, tag_to_idx)
    train_feats_int, val_feats_int = integerize(train_feats), integerize(val_feats)

    train_dataset, val_dataset = MyDataset(train_feats_int, train_feats_idx, train_labels), MyDataset(val_feats_int, val_feats_idx, val_labels)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, collate_fn=collate)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, collate_fn=collate)

    return train_loader, val_loader, word_to_idx

class Model(nn.Module):
    def __init__(self, word_to_idx):
        super(Model, self).__init__()
        self.conv1d = nn.Conv1d(1, CONV_OUT_CHANNEL, CONV_KERNEL_SIZE, padding=int((CONV_KERNEL_SIZE-1)/2))
        self.relu = nn.ReLU()
        self.word_embed = nn.Embedding(len(word_to_idx)+1, WORD_EMBEDDING_SIZE)
        self.dropout1 = nn.Dropout(p=DROPOUT)
        self.lstm = nn.LSTM(WORD_EMBEDDING_SIZE+CONV_OUT_CHANNEL, LSTM_FEATURE_SIZE, batch_first=True, num_layers=LSTM_LAYERS, bidirectional=True)
        self.dropout2 = nn.Dropout(p=DROPOUT)
        self.linear = nn.Linear(2*LSTM_FEATURE_SIZE, len(tags))
        self.padding_value = 0

    def forward(self, x_int, x_idx):
        x_int_data, x_int_batch_sizes = x_int.data, x_int.batch_sizes
        x_int_data = torch.unsqueeze(x_int_data, 1)
        x_int_data = self.conv1d(x_int_data)
        x_int_data = self.relu(x_int_data)
        x_int_data = torch.max(x_int_data, 2)[0]
        x_int = PackedSequence(x_int_data, x_int_batch_sizes)
        x_int_pad, lengths = pad_packed_sequence(x_int, batch_first=True, padding_value=self.padding_value)

        x_idx_pad, lengths = pad_packed_sequence(x_idx, batch_first=True, padding_value=self.padding_value)
        x_idx_pad = self.word_embed(x_idx_pad)

        x_pad = torch.cat([x_int_pad, x_idx_pad], 2)
        x_pad = self.dropout1(x_pad)

        x = pack_padded_sequence(x_pad, lengths, batch_first=True)
        h, _ = self.lstm(x)
        h_pad, lengths = pad_packed_sequence(h, batch_first=True, padding_value=self.padding_value)
        h_pad = self.dropout2(h_pad)

        h = pack_padded_sequence(h_pad, lengths, batch_first=True)
        x = h.data
        x = self.linear(x)

        return x

def train_val(num_epoch, dataloader_dict, model, loss, optimizer, print_epoch=True, device=None):
    val_acc_history = []
    val_loss_history = []
    train_acc_history = []
    train_loss_history = []
    best_acc = 0.0
    best_model_weights = copy.deepcopy(model.state_dict())
    
    for epoch in range(num_epoch):
        since = time.time()
        if print_epoch:
            print("Epoch {}/{}".format(epoch+1, num_epoch))
            print("-----------")
        
        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
            else :
                model.eval()
            total_loss = 0.0
            total_correct = 0.0
            num_words = 0
            for X_int, X_idx, y, batch_len in dataloader_dict[phase]:
                X_int = pack_sequence(X_int)
                X_idx = pack_sequence(X_idx)
                y = pack_sequence(y).data
                if device is not None:
                    X_int = X_int.to(device)
                    X_idx = X_idx.to(device)
                    y = y.to(device)
                with torch.set_grad_enabled(phase == "train"):
                    y_tilde = model(X_int, X_idx)
                    L = loss(y_tilde, y)
                    if phase == "train":
                        optimizer.zero_grad()
                        L.backward()
                        optimizer.step()
                y_tilde_label = torch.argmax(y_tilde, dim=1)
                num_correct = torch.sum((y_tilde_label == y))
                
                total_loss += L.item() * batch_len
                total_correct += num_correct.item()

                num_words += y.size()[0]
            
            epoch_loss = total_loss / len(dataloader_dict[phase].dataset)
            epoch_acc = (total_correct / num_words) * 100
            
            if phase == "val":
                val_loss_history.append(epoch_loss)
                val_acc_history.append(epoch_acc)
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_weights = copy.deepcopy(model.state_dict())
            if phase == "train":
                train_loss_history.append(epoch_loss)
                train_acc_history.append(epoch_acc)

            if print_epoch:
                print("{} | Loss : {:.4f}  Acc : {:.2f}%".format(phase.capitalize().ljust(5), epoch_loss, epoch_acc))
        
        print("Time elapsed : {:.2f} s".format(time.time() - since))
        print()
    
    model.load_state_dict(best_model_weights)
    return model, (val_acc_history, val_loss_history, train_acc_history, train_loss_history)

def train_model(train_file, model_file):
    # write your code here. You can add functions as well.
	# use torch library to save model parameters, hyperparameters, etc. to model_file

    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device : {}".format(dev))

    train_loader, val_loader, word_to_idx = build_dataset(train_file)

    model = Model(word_to_idx)
    for param in model.parameters():
        print(type(param.data), param.size())
    model.to(dev)
    loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    trained_model, history = train_val(EPOCH, {"train": train_loader, "val": val_loader}, model, loss, optimizer, device=dev)

    torch.save({
            'word_to_idx': word_to_idx,
            'model_state_dict': trained_model.state_dict(),
            }, model_file)
		
if __name__ == "__main__":
    # make no changes here
    train_file = sys.argv[1]
    model_file = sys.argv[2]
    train_model(train_file, model_file)
