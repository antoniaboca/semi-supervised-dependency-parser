import numpy as np
import pyconll
import pyconll.util
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence 
from torch.utils.data import Dataset

from allennlp.data.vocabulary import Vocabulary

class SentenceDataset(Dataset):
    def __init__(self, file, vocab=None, max_size=None, transform=None):
        self.transform = transform
        
        words = {}
        pos_tags = {}
        dep_rel = {}

        data = pyconll.load_from_file(file)
        
        if max_size is None:
            max_size = len(data)

        self.sentences = []
        self.parent_ids = []
        self.dep_indexes = []
        self.deps_list = []

        count = 0
        for sentence in data:
            word_list = []
            tag_list = []
            parents = []
            deps = []

            count += 1
            if count > max_size:
                break

            for token in sentence:
                if token.lemma is None:
                    continue
                if token.lemma in words:
                    words[token.lemma] += 1
                else:
                    words[token.lemma] = 1
                word_list.append(token.lemma)

                if token.upos is None:
                    continue

                if token.upos in pos_tags:
                    pos_tags[token.upos] += 1
                else:
                    pos_tags[token.upos] = 1
                tag_list.append(token.upos)

                if token.deprel is None:
                    continue
                if token.deprel in dep_rel:
                    dep_rel[token.deprel] += 1
                else:
                    dep_rel[token.deprel] = 1
                
                parents.append(token.head)
                deps.append(token.deprel)
                
            
            self.sentences.append((word_list, tag_list))
            self.parent_ids.append(parents)
            self.deps_list.append(deps)

        if vocab is None:
            self.vocabulary = Vocabulary(counter={'words': words, 'pos_tag': pos_tags, 'dep_rel': dep_rel})
        else:
            self.vocabulary = vocab

        self.index_to_word = self.vocabulary.get_index_to_token_vocabulary(namespace='words')
        self.index_to_pos = self.vocabulary.get_index_to_token_vocabulary(namespace='pos_tag')
        self.index_to_dep = self.vocabulary.get_index_to_token_vocabulary(namespace='dep_rel')

        self.word_to_index = self.vocabulary.get_token_to_index_vocabulary(namespace='words')
        self.pos_to_index = self.vocabulary.get_token_to_index_vocabulary(namespace='pos_tag')
        self.dep_to_index = self.vocabulary.get_token_to_index_vocabulary(namespace='dep_rel')

        self.index_set = []
        self.tag_set = []

        for sentence, tags in self.sentences:
            sidxs = [self.vocabulary.get_token_index(w, 'words') for w in sentence]
            tidxs = [self.vocabulary.get_token_index(t, 'pos_tag') for t in tags]

            self.index_set.append(sidxs)
            self.tag_set.append(tidxs)
        
        for deps in self.deps_list:
            didxs = [self.vocabulary.get_token_index(d, 'dep_rel') for d in deps]

            self.dep_indexes.append(didxs)

    def __len__(self):
        return len(self.index_set)
    
    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        sample = (self.index_set[index], self.tag_set[index])
        if self.transform:
            sample = self.transform(sample)
        
        return sample
    
    def getVocabulary(self):
        return self.vocabulary

class EmbeddingDataset(Dataset):
    def __init__(self, filename, dim, words_to_index):
        self.embeddings = {}
        self.dim = dim

        with open(filename) as file:
            for line in file:
                tokens = line.split()
                word = tokens[0]
                embeds = []
                for idx in range(1, dim + 1):
                    embeds.append(float(tokens[idx]))
                
                self.embeddings[word] = np.asarray(embeds)

        indexed = np.zeros((len(words_to_index), self.dim), dtype=float)
        for word, values in self.embeddings.items():
            if word in words_to_index:
                indexed[words_to_index[word]] = values
        
        self.idx_embeds = indexed
    
    def __len__(self):
        return len(self.idx_embeds)
    
    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        return self.idx_embeds[index]

class Embed(object):
    def __init__(self, embeddings):
        self.embeddings = embeddings
    
    def __call__(self, sample):
        sentence, tags = sample
        embedded = [self.embeddings[idx] for idx in sentence]

        return (sentence, embedded, tags)

def collate_fn_padder(samples):
    # batch of samples to be expected to look like
    # [(index_sent1, embed_sent1, tag_set1), ...]

    #import ipdb; ipdb.set_trace()
    indexes, embeds, tags = zip(*samples)

    sent_lens = torch.tensor([len(sent) for sent in indexes])

    indexes = [torch.tensor(sent) for sent in indexes]
    embeds = [torch.tensor(embed) for embed in embeds]
    tags = [torch.tensor(tag) for tag in tags]

    padded_sent = pad_sequence(indexes, batch_first=True)
    padded_embeds = pad_sequence(embeds, batch_first=True)
    padded_tags = pad_sequence(tags, batch_first=True)

    return {'sentence': padded_sent, 'embedding': padded_embeds, 'tags': padded_tags, 'lengths': sent_lens}


