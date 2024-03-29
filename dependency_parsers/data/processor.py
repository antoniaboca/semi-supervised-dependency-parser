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
    """PyTorch Dataset object that processes conllu files."""
    def __init__(self, file, ROOT_TOKEN, ROOT_TAG, ROOT_LABEL, vocab=None, transform=None, sentence_len=None):
        """
            The Constructor of the SentenceDataset class. The Constructor processes 
            conllu files to create datasets to work with.

        Args:
            file (str): Conllu file to load sentences from.
            ROOT_TOKEN (str): ROOT token to add at the beginning of each sentence.
            ROOT_TAG (str): ROOT tag to add at the beginning of each tag list.
            ROOT_LABEL (str): ROOT label to add at the beginning of each label list.
            vocab (allennlp.data.Vocabulary, optional): Prior vocabulary to use for
                words found in the dataset. Defaults to None.
            transform (func, optional): Transformation to apply to each element in the
                dataset. Defaults to None.
            sentence_len (int, optional): Maximum accepted length of sentence. Sentences
                with a greater length than this one will be skipped. Defaults to None.
        """
        self.transform = transform
        
        words = {}
        pos_tags = {}
        adv_tags = {}
        label = {}

        words[ROOT_TOKEN] = 0
        pos_tags[ROOT_TAG] = 0
        adv_tags[ROOT_TAG] = 0
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
            adv_list = [ROOT_TAG]
            parents = [0]
            labels = [ROOT_LABEL]

            words[ROOT_TOKEN] += 1
            pos_tags[ROOT_TAG] += 1
            adv_tags[ROOT_TAG] += 1
            label[ROOT_LABEL] += 1

            for token in sentence:
                if token.id.isdigit():
                    parents.append(int(token.head))
                else:
                    continue
                
                if token.upos is None or token.deprel is None:
                    continue
                
                if token.form is None and token.lemma is None:
                    continue

                if token.form is None:
                    word = token.lemma.lower()
                else:
                    word = token.form.lower()
                if word in words:
                    words[word] += 1
                else:
                    words[word] = 1
                word_list.append(word)

                if token.upos in pos_tags:
                    pos_tags[token.upos] += 1
                else:
                    pos_tags[token.upos] = 1
                tag_list.append(token.upos)

                if token.xpos is not None:
                    if token.xpos in adv_tags:
                        adv_tags[token.xpos] += 1
                    else:
                        adv_tags[token.xpos] = 1
                    adv_list.append(token.xpos)

                if token.deprel in label:
                    label[token.deprel] += 1
                else:
                    label[token.deprel] = 1
                
                labels.append(token.deprel)
                
            self.sentences.append((word_list, tag_list, adv_list, parents, labels))
            self.parent_ids.append(parents)

        if vocab is None:
            self.vocabulary = Vocabulary(
                counter={
                    'words': words, 
                    'pos_tag': pos_tags, 
                    'adv_tag': adv_tags,
                    'label': label
                    }
            )

        else:
            self.vocabulary = vocab

        self.index_to_word = self.vocabulary.get_index_to_token_vocabulary(namespace='words')
        self.index_to_pos = self.vocabulary.get_index_to_token_vocabulary(namespace='pos_tag')
        self.index_to_xpos = self.vocabulary.get_index_to_token_vocabulary(namespace='adv_tag')
        self.index_to_label = self.vocabulary.get_index_to_token_vocabulary(namespace='label')

        self.word_to_index = self.vocabulary.get_token_to_index_vocabulary(namespace='words')
        self.pos_to_index = self.vocabulary.get_token_to_index_vocabulary(namespace='pos_tag')
        self.xpos_to_index = self.vocabulary.get_token_to_index_vocabulary(namespace='adv_tag')
        self.label_to_index = self.vocabulary.get_token_to_index_vocabulary(namespace='label')

        self.index_set = []
        self.tag_set = []
        self.xpos_set = []
        self.parent_set = []
        self.label_set = []
        self.word_tag_set = []

        for sentence, tags, xpos, parents, labels in self.sentences:
            sidxs = [self.vocabulary.get_token_index(w, 'words') for w in sentence]
            tidxs = [self.vocabulary.get_token_index(t, 'pos_tag') for t in tags]
            xidxs = [self.vocabulary.get_token_index(x, 'adv_tag') for x in xpos]
            lidxs = [self.vocabulary.get_token_index(l, 'label') for l in labels]

            self.index_set.append(sidxs)
            self.tag_set.append(tidxs)
            self.xpos_set.append(xidxs)
            self.parent_set.append(parents)
            self.label_set.append(lidxs)
            self.word_tag_set.append(xpos)


    def __len__(self):
        """Return the number of available sentences.

        Returns:
            int: Length of sentence set.
        """
        return len(self.index_set)
    
    def __getitem__(self, index):
        """Returns a sentences (and its annotations) from the dataset.

        Args:
            index (int): The index of the sentence to fetch.

        Raises:
            AssertionError: Checks that all sentences contain the correct number
            of annotations.

        Returns:
            tuple: Word, upos, xpos, label indexes for each sentence and string XPOS tags.
        """
        if torch.is_tensor(index):
            index = index.tolist()

        sample = (
            self.index_set[index], 
            self.tag_set[index], 
            self.xpos_set[index],
            self.parent_set[index], 
            self.label_set[index],
            self.word_tag_set[index]
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
        """Returns the vocabulary used for this dataset."""
        return self.vocabulary

class EmbeddingDataset(Dataset):
    """Dataset object that processes .txt files that contain GloVe Embeddings."""
    def __init__(self, filename, dim, words_to_index, ROOT_TOKEN):
        """
        Constructor method for the embedding dataset. This constructor loads and 
        processes embeddings from files. The class removes embeddings of words that
        are not present in the vocabulary of the training set.

        Args:
            filename (str): file to load embeddings from.
            dim (int): Embedding dimension.
            words_to_index (dict): Dictionary with words as keys and their vocabulary
                indexes as values.
            ROOT_TOKEN (str): ROOT token to be used.
        """
        self.embeddings = {}
        self.dim = dim
        self.embeddings[ROOT_TOKEN] = np.random.rand(dim) # dummy random embedding for the root token

        with open(filename) as file:
            for line in file:
                tokens = line.split()
                word = tokens[0]
                embeds = []
                
                assert len(tokens) == dim + 1, "Dimension given does not match dimension in file"

                for idx in range(1, dim + 1):
                    embeds.append(float(tokens[idx]))
                
                self.embeddings[word] = np.asarray(embeds)

        indexed = np.zeros((len(words_to_index), self.dim), dtype=np.float64)
        
        self.missing = []
        for word, index in words_to_index.items():
            if word in self.embeddings:
                values = self.embeddings[word]
                indexed[index] = values
            else:
                self.missing.append(word)
        
        self.idx_embeds = indexed
    
    def __len__(self):
        """Returns the number of embeddings."""
        return len(self.idx_embeds)
    
    def __getitem__(self, index):
        """Return the embeddings of a particular word, given its index."""
        if torch.is_tensor(index):
            index = index.tolist()

        return self.idx_embeds[index]

class Embed(object):
    """Transformation object that embeds a given sentence."""
    def __init__(self, embeddings):
        self.embeddings = embeddings
    
    def __call__(self, sample):
        sentence, tags, xpos, parents = sample
        embedded = [self.embeddings[idx] for idx in sentence]

        return (sentence, embedded, tags, xpos, parents)

def labelled_padder(samples):
    """Padding function used to format the input to the supervised neural network.

    Args:
        samples (list): Batch of samples; each sample is a tuple expected to
            look like: (word indexes, UPOS idexes, XPOS indexes, 
            parent indexes, label indexes, string xpos tags)

    Returns:
        dict: Dictionary were each value is a list.
    """
    indexes, tags, xpos, parents, labels, word_tags = zip(*samples)

    sent_lens = torch.tensor([len(sent) for sent in indexes])

    indexes = [torch.tensor(sent) for sent in indexes]
    tags = [torch.tensor(tag) for tag in tags]
    xpos = [torch.tensor(pos) for pos in xpos]
    parents = [torch.tensor(parent) for parent in parents]
    labels = [torch.tensor(label) for label in labels]

    padded_sent = pad_sequence(indexes, batch_first=True, padding_value=0)
    padded_tags = pad_sequence(tags, batch_first=True, padding_value=0)
    padded_xpos = pad_sequence(xpos, batch_first=True, padding_value=0)
    padded_parents = pad_sequence(parents, batch_first=True, padding_value=-1)
    padded_labels = pad_sequence(labels, batch_first=True, padding_value=0)
    
    padded_parents[:, 0] = -1

    return {
        'sentence': padded_sent, 
        'tags': padded_tags,
        'xpos': padded_xpos,
        'parents': padded_parents, 
        'lengths': sent_lens,
        'labels': padded_labels,
        'word_tags': word_tags,
    }

def unlabelled_padder(samples):
    """Paddding function used to format the input to the semi-supervised neural network.

    Args:
        samples (list): Batch of samples; each sample is a tuple expected to look like:
            (word indexes, upos indexes, xpos indexes, labels, string xpos tags, features)

    Returns:
        dict: Dictionary were each value is a list.
    """

    indexes, tags, xpos, parents, labels, word_tags, features = zip(*samples)

    sent_lens = torch.tensor([len(sent) for sent in indexes])

    indexes = [torch.tensor(sent) for sent in indexes]
    tags = [torch.tensor(tag) for tag in tags]
    xpos = [torch.tensor(pos) for pos in xpos]

    parents = [torch.tensor(parent) for parent in parents]
    labels = [torch.tensor(label) for label in labels]

    padded_sent = pad_sequence(indexes, batch_first=True, padding_value=0)
    padded_tags = pad_sequence(tags, batch_first=True, padding_value=0)
    padded_xpos = pad_sequence(xpos, batch_first=True, padding_value=0)

    padded_parents = pad_sequence(parents, batch_first=True, padding_value=-1)
    padded_labels = pad_sequence(labels, batch_first=True, padding_value=0)

    maxlen = len(padded_sent[0])
    padded_features = [F.pad(torch.tensor(feature), (0, 0, 0, maxlen - feature.shape[-2], 0, maxlen - feature.shape[-3])) for feature in features]
    padded_features = torch.stack(padded_features)
    
    padded_parents[:, 0] = -1

    return {
        'sentence': padded_sent, 
        'upos': padded_tags, 
        'xpos': padded_xpos,
        'parents': padded_parents,
        'labels': padded_labels,
        'features': padded_features,
        'lengths': sent_lens,
        'word_tags': word_tags,
    }