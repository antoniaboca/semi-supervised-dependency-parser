import numpy as np
import pyconll
import pyconll.util
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence 
from torch.nn import functional as F

from torch.utils.data import Dataset

from allennlp.data.vocabulary import Vocabulary

PAD_VALUE = -100
PAD_IDX = 0

class SentenceDataset(Dataset):
    def __init__(self, file, ROOT_TOKEN, ROOT_TAG, ROOT_LABEL, vocab=None, transform=None, sentence_len=None):
        self.transform = transform
        
        words = {}
        pos_tags = {}
        label = {}

        words[ROOT_TOKEN] = 0
        pos_tags[ROOT_TAG] = 0
        label[ROOT_LABEL] = 0

        data = pyconll.load_from_file(file)

        self.sentences = []
        self.parent_ids = []
        self.dep_indexes = []

        for sentence in data:
            if sentence_len is not None and len(sentence) > sentence_len:
                continue
            
            word_list = [ROOT_TOKEN]
            tag_list = [ROOT_TAG]
            parents = [0]
            labels = [ROOT_LABEL]

            words[ROOT_TOKEN] += 1
            pos_tags[ROOT_TAG] += 1
            label[ROOT_LABEL] += 1

            for token in sentence:
                if token.id.isdigit():
                    parents.append(int(token.head))
                else:
                    continue

                if token.lemma is None or token.upos is None or token.deprel is None:
                    continue

                word = token.lemma.lower()
                if token.lemma in words:
                    words[word] += 1
                else:
                    words[word] = 1
                word_list.append(word)

                if token.upos in pos_tags:
                    pos_tags[token.upos] += 1
                else:
                    pos_tags[token.upos] = 1
                tag_list.append(token.upos)

                if token.deprel in label:
                    label[token.deprel] += 1
                else:
                    label[token.deprel] = 1
                
                labels.append(token.deprel)
                
            
            self.sentences.append((word_list, tag_list, parents, labels))
            self.parent_ids.append(parents)

        if vocab is None:
            self.vocabulary = Vocabulary(
                counter={
                    'words': words, 
                    'pos_tag': pos_tags, 
                    'label': label
                    }
            )

        else:
            self.vocabulary = vocab

        self.index_to_word = self.vocabulary.get_index_to_token_vocabulary(namespace='words')
        self.index_to_pos = self.vocabulary.get_index_to_token_vocabulary(namespace='pos_tag')
        self.index_to_label = self.vocabulary.get_index_to_token_vocabulary(namespace='label')

        self.word_to_index = self.vocabulary.get_token_to_index_vocabulary(namespace='words')
        self.pos_to_index = self.vocabulary.get_token_to_index_vocabulary(namespace='pos_tag')
        self.label_to_index = self.vocabulary.get_token_to_index_vocabulary(namespace='label')

        self.index_set = []
        self.tag_set = []
        self.parent_set = []
        self.label_set = []

        for sentence, tags, parents, labels in self.sentences:
            sidxs = [self.vocabulary.get_token_index(w, 'words') for w in sentence]
            tidxs = [self.vocabulary.get_token_index(t, 'pos_tag') for t in tags]
            lidxs = [self.vocabulary.get_token_index(l, 'label') for l in labels]

            self.index_set.append(sidxs)
            self.tag_set.append(tidxs)
            self.parent_set.append(parents)
            self.label_set.append(lidxs)

    def __len__(self):
        return len(self.index_set)
    
    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        sample = (
            self.index_set[index], 
            self.tag_set[index], 
            self.parent_set[index], 
            self.label_set[index]
        )

        if self.transform:
            sample = self.transform(sample)
        
        try:
            assert len(self.index_set[index]) == len(self.parent_set[index])
        except AssertionError:
            print(index)
            print('Length {} for {}'.format(len(self.sentences[index][0]), self.sentences[index][0]))
            print('Length {} for {}'.format(len(self.index_set[index]), self.index_set[index]))
            print('Length {} for {}'.format(len(self.parent_set[index]), self.parent_set[index]))
            print('Length {} for {}'.format(len(self.label_set[index]), self.label_set[index]))
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

        #indexed = np.random.rand(len(words_to_index), self.dim)
        indexed = np.zeros((len(words_to_index), self.dim), dtype=np.float64)

        #for word, values in self.embeddings.items():
        #    if word in words_to_index:
        #        indexed[words_to_index[word]] = values
        #        count -= 1

        missing = []
        for word, index in words_to_index.items():
            if word in self.embeddings:
                values = self.embeddings[word]
                indexed[index] = values
            else:
                missing.append(word)
        
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

def labelled_padder(samples):
    # batch of samples to be expected to look like
    # [(index_sent1, tag_set1, parent_set1, label_set1), ...]

    #import ipdb; ipdb.set_trace()
    indexes, tags, parents, labels = zip(*samples)

    sent_lens = torch.tensor([len(sent) for sent in indexes])

    indexes = [torch.tensor(sent) for sent in indexes]
    tags = [torch.tensor(tag) for tag in tags]
    parents = [torch.tensor(parent) for parent in parents]
    labels = [torch.tensor(label) for label in labels]

    padded_sent = pad_sequence(indexes, batch_first=True, padding_value=0)
    padded_tags = pad_sequence(tags, batch_first=True, padding_value=0)
    padded_parents = pad_sequence(parents, batch_first=True, padding_value=-1)
    padded_labels = pad_sequence(labels, batch_first=True, padding_value=0)
    
    padded_parents[:, 0] = -1

    return {
        'sentence': padded_sent, 
        'tags': padded_tags, 
        'parents': padded_parents, 
        'lengths': sent_lens,
        'labels': padded_labels,
    }

def unlabelled_padder(samples):
    # batch of samples to be expected to look like
    # [(idxs1, tags1, features1), ...]

    indexes, tags, parents, labels, features = zip(*samples)

    sent_lens = torch.tensor([len(sent) for sent in indexes])

    indexes = [torch.tensor(sent) for sent in indexes]
    tags = [torch.tensor(tag) for tag in tags]

    parents = [torch.tensor(parent) for parent in parents]
    labels = [torch.tensor(label) for label in labels]

    padded_sent = pad_sequence(indexes, batch_first=True, padding_value=0)
    padded_tags = pad_sequence(tags, batch_first=True, padding_value=0)

    padded_parents = pad_sequence(parents, batch_first=True, padding_value=-1)
    padded_labels = pad_sequence(labels, batch_first=True, padding_value=0)

    maxlen = len(padded_sent[0])
    padded_features = [F.pad(torch.tensor(feature), (0, 0, 0, maxlen - feature.shape[-2], 0, maxlen - feature.shape[-3])) for feature in features]
    padded_features = torch.stack(padded_features)

    padded_parents[:, 0] = -1

    return {
        'sentence': padded_sent, 
        'tags': padded_tags, 
        'parents': padded_parents,
        'labels': padded_labels,
        'features': padded_features,
        'lengths': sent_lens,
    }