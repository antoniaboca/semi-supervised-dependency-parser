import pickle
import numpy as np
from .data.processor import collate_fn_padder

import pytorch_lightning as pl
from torch.utils.data import DataLoader


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
        assert len(unlabelled_set[0]) == 2 # I assume an unlabelled tuple has word indexes and pos tag indexes
        
    feature_list = []
    for idxs, tags in unlabelled_set:
        vec = np.zeros((len(tags), len(tags), len(order)), dtype=np.int32)
        for idx1 in range(len(tags)):
            for idx2 in range(len(tags)):
                assert tags[idx1] < 20 and tags[idx2] < 20
                
                if (tags[idx1], tags[idx2]) in order:
                    vec[idx1][idx2][order[(tags[idx1], tags[idx2])]] = 1
        feature_list.append((idxs, tags, vec))
    
    return feature_list

class DataModule(pl.LightningDataModule):
    def __init__(self, PICKLE_FILE, BATCH_SIZE, EMBEDDING_DIM,
                TRAIN_SIZE, VAL_SIZE, TEST_SIZE, args):

        with open(PICKLE_FILE, 'rb') as file:
            object = pickle.load(file)
            if args.semi:
                train_set = object['train_unlabelled'][:TRAIN_SIZE]
                self.labelled = object['train_labelled'][:args.labelled_size]

                print('Creating prior distribution for the semi-supervised context...')
                tree_edges, graph_edges = edge_count(self.labelled)
                self.features20, self.order20 = top20(tree_edges, graph_edges)
                feature_set = feature_vector(self.order20, train_set) # this is the training set for the semi-supervised context

            else:
                train_set = object['train_labelled'][:TRAIN_SIZE]

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