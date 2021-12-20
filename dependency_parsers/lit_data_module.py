from os import X_OK
import pickle

from .data.processor import collate_fn_padder

import pytorch_lightning as pl
from torch.utils.data import DataLoader

class DataModule(pl.LightningDataModule):
    def __init__(self, PICKLE_FILE, BATCH_SIZE, EMBEDDING_DIM,
                TRAIN_SIZE, VAL_SIZE, TEST_SIZE):

        with open(PICKLE_FILE, 'rb') as file:
            object = pickle.load(file)
            train_set = object['train'][:TRAIN_SIZE]
            dev_set = object['dev'][:VAL_SIZE]
            test_set = object['test'][:TEST_SIZE]
            self.embeddings = object['embeddings']
            self.TAGSET_SIZE = object['TAGSET_SIZE']
            self.LABSET_SIZE = object['LABSET_SIZE']

            assert self.embeddings.shape[-1] == EMBEDDING_DIM, "The embedding dimension does not match the loaded embedding file"

        self.train_dataloader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn_padder)
        self.test_dataloader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn_padder)
        self.dev_dataloader = DataLoader(dev_set, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn_padder)

    def dev_dataloader(self):
        return self.dev_dataloader

    def train_dataloader(self):
        return self.train_dataloader

    def test_dataloader(self):
        return self.test_dataloader