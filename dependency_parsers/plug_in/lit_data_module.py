import pickle
import torch

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence

def collate_fn_padder(samples):
    # batch of samples to be expected to look like
    # [(index_sent1, embed_sent1, tag_set1, parent_set1), ...]

    #import ipdb; ipdb.set_trace()
    indexes, tags, parents = zip(*samples)

    # indexes, embeds, tags, parents = zip(*samples)

    sent_lens = torch.tensor([len(sent) for sent in indexes])

    indexes = [torch.tensor(sent) for sent in indexes]
    # embeds = [torch.tensor(embed) for embed in embeds]
    tags = [torch.tensor(tag) for tag in tags]
    parents = [torch.tensor(parent) for parent in parents]

    padded_sent = pad_sequence(indexes, batch_first=True, padding_value=0)
    # padded_embeds = pad_sequence(embeds, batch_first=True, padding_value=0.0)
    padded_tags = pad_sequence(tags, batch_first=True, padding_value=0)
    padded_parents = pad_sequence(parents, batch_first=True, padding_value=-100)

    #assert len(padded_parents[0]) == len(padded_embeds[0])

    return {
        'sentence': padded_sent, 
        #'embedding': padded_embeds, 
        'tags': padded_tags, 
        'parents': padded_parents, 
        'lengths': sent_lens
    }

def sorted(x):
    return len(x[0])

class DataModule(pl.LightningDataModule):
    def __init__(self, PICKLE_FILE, BATCH_SIZE, PARAM_FILE):
        with open(PICKLE_FILE, 'rb') as file:
            object = pickle.load(file)
            self.train_set = object['train']
            self.dev_set = object['dev']
            self.test_set = object['test']
            self.embeddings = object['embeddings']

        self.train_set.sort(key=sorted)
        self.train_dataloader = DataLoader(self.train_set, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn_padder)
        self.test_dataloader = DataLoader(self.test_set, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn_padder)
        self.dev_dataloader = DataLoader(self.dev_set, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn_padder)

        with open(PARAM_FILE, 'rb') as file:
            dict = pickle.load(file)
        self.TAGSET_SIZE = dict['TAGSET_SIZE']

    def dev_dataloader(self):
        return self.dev_dataloader

    def train_dataloader(self):
        return self.train_dataloader

    def test_dataloader(self):
        return self.test_dataloader