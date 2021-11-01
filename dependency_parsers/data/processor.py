import numpy as np
import pyconll
import pyconll.util
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence 
from torch.utils.data import Dataset

from allennlp.data.vocabulary import Vocabulary

PAD_VALUE = -100

class SentenceDataset(Dataset):
    def __init__(self, file, ROOT_TOKEN, ROOT_TAG, ROOT_LABEL, vocab=None, max_size=None, transform=None):
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
            word_list = [ROOT_TOKEN]
            tag_list = [ROOT_TAG]
            parents = [0]
            deps = [ROOT_LABEL]

            count += 1
            if count > max_size:
                break

            for token in sentence:
                if token.id.isdigit():
                    parents.append(int(token.head))
                else:
                    continue

                if token.lemma is None or token.upos is None or token.deprel is None:
                    continue

                if token.lemma in words:
                    words[token.lemma] += 1
                else:
                    words[token.lemma] = 1
                word_list.append(token.lemma)

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
                
                deps.append(token.deprel)
                
            
            self.sentences.append((word_list, tag_list, parents))
            self.parent_ids.append(parents)
            self.deps_list.append(deps)

        if vocab is None:
            self.vocabulary = Vocabulary(
                counter={
                    'words': words, 
                    'pos_tag': pos_tags, 
                    'dep_rel': dep_rel
                    },
                tokens_to_add={
                    'words': [ROOT_TOKEN], 
                    'pos_tag': [ROOT_TAG],
                    'dep_rel': [ROOT_LABEL]
                    }
            )

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
        self.parent_set = []

        for sentence, tags, parents in self.sentences:
            sidxs = [self.vocabulary.get_token_index(w, 'words') for w in sentence]
            tidxs = [self.vocabulary.get_token_index(t, 'pos_tag') for t in tags]

            self.index_set.append(sidxs)
            self.tag_set.append(tidxs)
            self.parent_set.append(parents)
        
        for deps in self.deps_list:
            didxs = [self.vocabulary.get_token_index(d, 'dep_rel') for d in deps]

            self.dep_indexes.append(didxs)

    def __len__(self):
        return len(self.index_set)
    
    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        sample = (self.index_set[index], self.tag_set[index], self.parent_set[index])
        if self.transform:
            sample = self.transform(sample)
        
        try:
            assert len(self.index_set[index]) == len(self.parent_set[index])
        except AssertionError:
            print(index)
            print('Length {} for {}'.format(len(self.sentences[index][0]), self.sentences[index][0]))
            print('Length {} for {}'.format(len(self.index_set[index]), self.index_set[index]))
            print('Length {} for {}'.format(len(self.parent_set[index]), self.parent_set[index]))
            raise AssertionError

        return sample
    
    def getVocabulary(self):
        return self.vocabulary

class EmbeddingDataset(Dataset):
    def __init__(self, filename, dim, words_to_index, ROOT_TOKEN):
        self.embeddings = {}
        self.dim = dim
        self.embeddings[ROOT_TOKEN] = np.random.rand(dim) # dummy random embedding for the root token

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
        sentence, tags, parents = sample
        embedded = [self.embeddings[idx] for idx in sentence]

        return (sentence, embedded, tags, parents)

def collate_fn_padder(samples):
    # batch of samples to be expected to look like
    # [(index_sent1, embed_sent1, tag_set1, parent_set1), ...]

    #import ipdb; ipdb.set_trace()
    indexes, embeds, tags, parents = zip(*samples)

    sent_lens = torch.tensor([len(sent) for sent in indexes])

    indexes = [torch.tensor(sent) for sent in indexes]
    embeds = [torch.tensor(embed) for embed in embeds]
    tags = [torch.tensor(tag) for tag in tags]
    parents = [torch.tensor(parent) for parent in parents]

    padded_sent = pad_sequence(indexes, batch_first=True, padding_value=PAD_VALUE)
    padded_embeds = pad_sequence(embeds, batch_first=True, padding_value=PAD_VALUE)
    padded_tags = pad_sequence(tags, batch_first=True, padding_value=PAD_VALUE)
    padded_parents = pad_sequence(parents, batch_first=True, padding_value=PAD_VALUE)

    assert len(padded_parents[0]) == len(padded_embeds[0])

    return {
        'sentence': padded_sent, 
        'embedding': padded_embeds, 
        'tags': padded_tags, 
        'parents': padded_parents, 
        'lengths': sent_lens
    }


