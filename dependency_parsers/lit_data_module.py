import pickle
import numpy as np
from .data.processor import labelled_padder, unlabelled_padder

import pytorch_lightning as pl
from torch.utils.data import DataLoader

import torch

def edge_count(set):
    tree_edges = {}
    graph_edges = {}

    for sentence in set:
        for idx1 in range(len(sentence[0])):
            for idx2 in range(len(sentence[0])):
                if idx1 == idx2:
                    continue

                tag1 = sentence[1][idx1]
                tag2, parent2 = sentence[1][idx2], sentence[2][idx2]
                
                if (tag1, tag2) not in graph_edges:
                    graph_edges[(tag1, tag2)] = 0
                graph_edges[(tag1, tag2)] += 1

                if parent2 == idx1:
                    if (tag1, tag2) not in tree_edges:
                        tree_edges[(tag1, tag2)] = 0
                    tree_edges[(tag1, tag2)] += 1

    return tree_edges, graph_edges

def top20(tree, graph):
    distribution = {}
    order = {}
    for edge in graph.keys():
        if graph[edge] < 10:
            continue
        distribution[edge] = tree.get(edge, 0) / graph[edge]
    top = []
    for key, value in distribution.items():
        top.append((value, key))
    top.sort(reverse=True)
    distribution = {}

    idx = 0
    for value, key in top[:20]:
        distribution[key] = value
        order[key] = idx
        idx += 1
        
    return distribution, order

def feature_vector(order, unlabelled_set):
    if len(unlabelled_set) > 0:
        assert len(unlabelled_set[0]) == 4 # I assume an unlabelled tuple has word indexes, tags, parents, labels and feature vectors
        
    feature_list = []
    for idxs, tags, _, _ in unlabelled_set:
        vec = np.zeros((len(tags), len(tags), len(order)), dtype=np.int32)
        for idx1 in range(len(tags)):
            for idx2 in range(len(tags)):
                assert tags[idx1] < 20 and tags[idx2] < 20

                if (tags[idx1], tags[idx2]) in order:
                    vec[idx1][idx2][order[(tags[idx1], tags[idx2])]] = 1
        feature_list.append(vec)
    
    return feature_list

class DataModule(pl.LightningDataModule):
    def __init__(self, PICKLE_FILE, BATCH_SIZE, EMBEDDING_DIM,
                TRAIN_SIZE, VAL_SIZE, TEST_SIZE, args):
        super().__init__()

        self.semi = args.semi
        self.TRAIN_SIZE = TRAIN_SIZE
        self.labelled_size = args.labelled_size
        self.VAL_SIZE = VAL_SIZE
        self.TEST_SIZE = TEST_SIZE
        self.PICKLE_FILE = PICKLE_FILE
        self.EMBEDDING_DIM = EMBEDDING_DIM
        self.BATCH_SIZE = BATCH_SIZE

    def prepare_data(self):
        with open(self.PICKLE_FILE, 'rb') as file:
            object = pickle.load(file)
            if self.semi:
                self.unlabelled_set = object['train_unlabelled'][:self.TRAIN_SIZE]
                self.labelled_set = object['train_labelled'][:self.labelled_size]
            else:
                self.labelled_set = object['train_labelled'][:self.TRAIN_SIZE]
    
            self.dev_set = object['dev'][:self.VAL_SIZE]
            self.test_set = object['test'][:self.TEST_SIZE]
            self.embeddings = object['embeddings']
            self.TAGSET_SIZE = object['TAGSET_SIZE']
            self.LABSET_SIZE = object['LABSET_SIZE']

            assert self.embeddings.shape[-1] == self.EMBEDDING_DIM, "The embedding dimension does not match the loaded embedding file"

    def setup(self, stage):
        if self.semi:
            if stage in (None, 'fit'):
                print('Creating prior distribution for the semi-supervised context...')
                tree_edges, graph_edges = edge_count(self.labelled_set)
                self.features20, self.order20 = top20(tree_edges, graph_edges)
                feature_set = feature_vector(self.order20, self.unlabelled_set) # this is the training set for the semi-supervised context
                    
                self.unlabelled_set = [(*a,b) for a, b in zip(self.unlabelled_set, feature_set)]
    
    def train_dataloader(self):
        if not self.semi:
            return DataLoader(self.labelled_set, batch_size=self.BATCH_SIZE, shuffle=False, collate_fn=labelled_padder)
        
        labelled = DataLoader(self.labelled_set, batch_size=self.labelled_size, shuffle=False, collate_fn=labelled_padder)
        unlabelled = DataLoader(self.unlabelled_set, batch_size=self.BATCH_SIZE, shuffle=False, collate_fn=unlabelled_padder)

        return {'labelled': labelled, 'unlabelled': unlabelled}

    def val_dataloader(self):
        return DataLoader(self.dev_set, batch_size=self.BATCH_SIZE, shuffle=False, collate_fn=labelled_padder)
  
    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.BATCH_SIZE, shuffle=False, collate_fn=labelled_padder)

    def get_prior(self):
        vector20 = torch.zeros(20)
        for key, value in self.order20.items():
            vector20[value] = self.features20[key]

        return vector20

    