import pickle

import torchmetrics

from data.processor import collate_fn_padder

import pytorch_lightning as pl
from torch.utils.data import DataLoader

class DataModule(pl.LightningDataModule):
    def __init__(self, PICKLE_FILE, BATCH_SIZE, PARAM_FILE):
        with open(PICKLE_FILE, 'rb') as file:
            object = pickle.load(file)
            train_set = object['train']
            dev_set = object['dev']
            test_set = object['test']

        self.train_dataloader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn_padder)
        self.test_dataloader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn_padder)
        self.dev_dataloader = DataLoader(dev_set, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn_padder)

        with open(PARAM_FILE, 'rb') as file:
            dict = pickle.load(file)
        self.TAGSET_SIZE = dict['TAGSET_SIZE']

    def dev_dataloader(self):
        return self.dev_dataloader

    def train_dataloader(self):
        return self.train_dataloader

    def test_dataloader(self):
        return self.test_dataloader