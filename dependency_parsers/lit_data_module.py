import pickle
from unicodedata import name
import numpy as np

from dependency_parsers.analysis.counter import TagCounter
from .data.processor import labelled_padder, unlabelled_padder
from .data.oracle import order20, features20
from .analysis.counter import TagCounter

import pytorch_lightning as pl
from torch.utils.data import DataLoader

import torch

def feature_vector(order, unlabelled_set, tag_type):
    if len(unlabelled_set) > 0:
        assert len(unlabelled_set[0]) == 6 # I assume an unlabelled tuple has word indexes, tags, xpos, parents, labels and feature vectors
        
    feature_list = []
    for idxs, upos, xpos, _, _, _ in unlabelled_set:
        if tag_type == 'xpos':
            pos = xpos
        else:
            pos = upos
        vec = np.zeros((len(pos), len(pos), len(order)), dtype=np.int32)
        for idx1 in range(len(pos)):
            for idx2 in range(len(pos)):
                # assert xpos[idx1] < 20 and xpos[idx2] < 20
                if idx1 == idx2:
                    continue

                if (pos[idx1], pos[idx2]) in order:
                    vec[idx1][idx2][order[(pos[idx1], pos[idx2])]] = 1
        feature_list.append(vec)
    
    return feature_list

class DataModule(pl.LightningDataModule):
    def __init__(self, PICKLE_FILE, BATCH_SIZE, EMBEDDING_DIM,
                TRAIN_SIZE, VAL_SIZE, TEST_SIZE, args):
        super().__init__()

        self.transfer = args.transfer
        self.entropy = args.entropy
        self.semi = args.semi
        self.semi_labelled_batch = args.semi_labelled_batch
        self.TRAIN_SIZE = TRAIN_SIZE
        self.labelled_size = args.labelled_size
        self.VAL_SIZE = VAL_SIZE
        self.TEST_SIZE = TEST_SIZE
        self.PICKLE_FILE = PICKLE_FILE
        self.EMBEDDING_DIM = EMBEDDING_DIM
        self.BATCH_SIZE = BATCH_SIZE
        self.oracle = args.oracle
        self.ge_only = args.ge_only
        self.tag_type = args.tag_type
        
        if args.limit_sentence_size > 0:
            self.filter = lambda x: len(x[0]) < args.limit_sentence_size
        else:
            self.filter = lambda x: True

    def prepare_data(self):
        with open(self.PICKLE_FILE, 'rb') as file:
            object = pickle.load(file)
            if self.transfer:
                total = list(filter(self.filter, object['remainder']))
                self.labelled_set = total[:self.labelled_size]
                self.unlabelled_set = total[self.labelled_size : self.labelled_size + self.TRAIN_SIZE]
            else:
                if self.entropy:
                    self.unlabelled_set = list(filter(self.filter, object['remainder']))[:self.TRAIN_SIZE]
                    self.labelled_set = object['train_labelled'][:self.labelled_size]
                elif self.semi:
                    self.unlabelled_set = list(filter(self.filter, object['train_unlabelled']))[:self.TRAIN_SIZE]
                    self.labelled_set = object['train_labelled'][:self.labelled_size]
                else:
                    self.labelled_set = list(filter(self.filter, object['train_labelled']))[:self.TRAIN_SIZE]

            self.dev_set = list(filter(self.filter, object['dev']))[:self.VAL_SIZE]
            self.test_set = list(filter(self.filter, object['test']))[:self.TEST_SIZE]
            
            #self.dev_set = object['dev'][:self.VAL_SIZE]
            #self.test_set = object['test'][:self.TEST_SIZE]
    
            self.embeddings = object['embeddings']
            self.TAGSET_SIZE = object['TAGSET_SIZE']
            self.LABSET_SIZE = object['LABSET_SIZE']
            self.vocabulary = object['vocabulary']

            assert self.embeddings.shape[-1] == self.EMBEDDING_DIM, "The embedding dimension does not match the loaded embedding file"

    def setup(self, stage):
        if self.semi:
            if self.tag_type is 'xpos':
                namespc = 'adv_tag'
            else:
                namespc = 'pos_tag'

            if stage in (None, 'fit'):
                print('Creating prior distribution for the semi-supervised context...')
                counter = TagCounter(sentence_set=self.labelled_set)

                if not self.oracle:
                    tree_edges, graph_edges = counter.edge_count(self.tag_type)
                    self.features20, self.order20 = TagCounter.top_edges(tree_edges, graph_edges, 20, 100)
                else:
                    self.features20, self.order20 = {}, {}
                    
                    for key, value in features20.items():
                        idx_key = (self.vocabulary.get_token_index(key[0], namespace=namespc), self.vocabulary.get_token_index(key[1], namespace=namespc))
                        self.features20[idx_key] = value
                        self.order20[idx_key] = order20[key]
                    
                oracle = 0
                totals = 0
                for _, upos, xpos, parents, _, word_tags in self.unlabelled_set:
                    for idx in range(1, len(parents)):
                        xhead = xpos[parents[idx]]
                        xdep = xpos[idx]

                        uhead = upos[parents[idx]]
                        udep = upos[idx]

                        try:
                            assert xpos[idx] == self.vocabulary.get_token_index(word_tags[idx], namespace='adv_tag')
                            assert self.vocabulary.get_token_from_index(xpos[idx], namespace='adv_tag') == word_tags[idx]
                        except:
                            import ipdb; ipdb.post_mortem()

                        if namespc == 'xpos':
                            head, dep = xhead, xdep
                        else:
                            head, dep = uhead, udep

                        if (head,dep) in self.features20:
                            oracle += 1
                        totals += 1
                print('\n Percentage of target edges that are oracle edges: {:3.3f}\n'.format(oracle/totals))
                    
                feature_set = feature_vector(self.order20, self.unlabelled_set, self.tag_type) # this is the training set for the semi-supervised context
                self.unlabelled_set = [(*a,b) for a, b in zip(self.unlabelled_set, feature_set)]

                feature_val = feature_vector(self.order20, self.dev_set, self.tag_type)                
                self.dev_set = [(*a, b) for a, b in zip(self.dev_set, feature_val)]
    
    def train_dataloader(self):
        if not self.semi:
            return DataLoader(self.labelled_set, batch_size=self.BATCH_SIZE, shuffle=False, collate_fn=labelled_padder)
        
        if not self.ge_only:
            labelled = DataLoader(self.labelled_set, batch_size=self.semi_labelled_batch, shuffle=False, collate_fn=labelled_padder)
            unlabelled = DataLoader(self.unlabelled_set, batch_size=self.BATCH_SIZE, shuffle=False, collate_fn=unlabelled_padder)
            return {'labelled': labelled, 'unlabelled': unlabelled}
        
        unlabelled = DataLoader(self.unlabelled_set, batch_size=self.BATCH_SIZE, shuffle=False, collate_fn=unlabelled_padder)
        return {'unlabelled': unlabelled}

    def val_dataloader(self):
        if not self.semi:
            return DataLoader(self.dev_set, batch_size=self.BATCH_SIZE, shuffle=False, collate_fn=labelled_padder)
        return DataLoader(self.dev_set, batch_size=self.BATCH_SIZE, shuffle=False, collate_fn=unlabelled_padder)
  
    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.BATCH_SIZE, shuffle=False, collate_fn=labelled_padder)

    def get_prior(self):
        vector20 = torch.zeros(20)
        for key, value in self.order20.items():
            vector20[value] = self.features20[key]

        return vector20

    