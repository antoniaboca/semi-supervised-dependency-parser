import pickle
import importlib

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_packed_sequence
from torch.nn import Linear, LSTM
from torch.utils.data.dataset import random_split
from lstm import LSTMTagger

from data.processor import collate_fn_padder
import pytorch_lightning as pl

class LitLSTMTagger(pl.LightningModule):
    def __init__(self, embedding_dim, hidden_dim, num_layers, dropout, tagset_size):
        super(LitLSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim

        self.lstm = LSTM(
            input_size=embedding_dim, 
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=True
        )

        self.hidden2tag = Linear(hidden_dim * 2, tagset_size)

        self.loss_function = nn.CrossEntropyLoss(ignore_index=0)

    def forward(self, x):
        embedding = x['embedding']
        
        lstm_out, _ = self.lstm(embedding.float())
        lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)
        
        tag_scores = self.hidden2tag(lstm_out)
        return tag_scores
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=0.05)
        return optimizer
    
    def training_step(self, train_batch, batch_idx):
        targets, _ = pad_packed_sequence(train_batch['tags'], batch_first=True)

        tag_scores = self(train_batch)

        batch_size, sent_len, num_tags = tag_scores.shape
        total_loss = self.loss_function(
            tag_scores.reshape(batch_size * sent_len, num_tags),
            targets.reshape(batch_size * sent_len)
        )

        self.log("train_loss", total_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return total_loss

class DataModule(pl.LightningDataModule):
    def __init__(self, PICKLE_FILE, BATCH_SIZE, TRAIN_SPLIT, VAL_SPLIT, PARAM_FILE):
        with open(PICKLE_FILE, 'rb') as file:
            embedded_set = pickle.load(file)
        train, val = random_split(embedded_set, [TRAIN_SPLIT, VAL_SPLIT])
        
        self.train_dataloader = DataLoader(train, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn_padder)
        self.val_dataloader = DataLoader(val, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn_padder)

        with open(PARAM_FILE, 'rb') as file:
            dict = pickle.load(file)
        self.TAGSET_SIZE = dict['TAGSET_SIZE']

    def train_dataloader(self):
        return self.train_dataloader

    def val_dataloader(self):
        return self.val_dataloader