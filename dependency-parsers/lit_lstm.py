import pickle
import importlib

import torchmetrics

import torch
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

        self.loss_fn = []

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

        tags = tag_scores.argmax(-1)

        num_correct = 0
        total = 0

        num_correct += torch.count_nonzero((tags == targets) * (targets != 0))
        total += torch.count_nonzero(targets)

        #self.log("train_loss", total_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return {'loss': total_loss, 'correct': num_correct, 'total': total}

    def training_epoch_end(self, outputs):
        correct = 0
        total = 0
        for output in outputs:
            correct += output['correct']
            total += output['total']
            self.loss_fn.append(output['loss'])
        
        print('Accuracy after epoch end: {}'.format(correct/total))

    def validation_step(self, batch, batch_idx):
        targets, _ = pad_packed_sequence(batch['tags'], batch_first=True)

        tag_scores = self(batch)

        batch_size, sent_len, num_tags = tag_scores.shape
        total_loss = self.loss_function(
            tag_scores.reshape(batch_size * sent_len, num_tags),
            targets.reshape(batch_size * sent_len)
        )

        tags = tag_scores.argmax(-1)

        num_correct = 0
        total = 0

        num_correct += torch.count_nonzero((tags == targets) * (targets != 0))
        total += torch.count_nonzero(targets)

        #self.log("train_loss", total_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return {'loss': total_loss, 'correct': num_correct, 'total': total}

    def validation_epoch_end(self, preds):
        correct = 0
        total = 0
        loss = 0

        for pred in preds:
            correct += pred['correct']
            total += pred['total']
            loss += pred['loss']

        self.log('accuracy', correct / total)
        self.log('loss', loss/len(preds))

        return {'accuracy': correct / total, 'loss': loss/len(preds)}

    def test_step(self, batch, batch_idx):
        return dict(self.validation_step(batch, batch_idx))
    
    def test_epoch_end(self, preds):
        return self.validation_epoch_end(preds)

class DataModule(pl.LightningDataModule):
    def __init__(self, PICKLE_FILE, BATCH_SIZE, PARAM_FILE):
        with open(PICKLE_FILE, 'rb') as file:
            object = pickle.load(file)
            train_set = object['train']
            dev_set = object['dev']
            test_set = object['test']

        self.train_dataloader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn_padder)
        self.test_dataloader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn_padder)
        self.dev_dataloader = DataLoader(dev_set, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn_padder)

        with open(PARAM_FILE, 'rb') as file:
            dict = pickle.load(file)
        self.TAGSET_SIZE = dict['TAGSET_SIZE']

    def dev_dataloader(self):
        return self.dev_dataloader

    def train_dataloader(self):
        return self.train_dataloader

    def test_dataloader(self):
        return self.test_dataloader