from logging import root
import numpy as np
import unittest
import torch

from allennlp.data.vocabulary import Vocabulary
from dependency_parsers.data.params import ROOT_TAG, ROOT_TOKEN, ROOT_LABEL
from dependency_parsers.data.processor import SentenceDataset, EmbeddingDataset, labelled_padder, unlabelled_padder

class DatasetTestCase(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.file = 'tests/test_vocabulary.conllu'
        self.embedding_file = 'tests/test_embedding.txt'
        self.data_file = 'tests/test_data.conllu'

        words = {
            ROOT_TOKEN: 1,
            'this': 2,
            'is': 1,
            'a': 1,
            'test': 1,
            'for': 1,
            'vocabulary': 1,
            '.': 1}

        tags = {
            ROOT_TAG: 1,
            'PRON': 2,
            'VERB': 1,
            'DET': 1,
            'NOUN': 2,
            'ADP': 1,
            'PUNCT': 1,
        }

        labels = {
            ROOT_LABEL: 1,
            'nsubj': 2,
            'conj': 1,
            'det': 1,
            'compound': 1,
            'case': 1,
            'obj': 1,
            'punct': 1,
        }

        xpos = {
            ROOT_TAG: 1,
            'DT': 3,
            'VBZ': 1,
            'NN': 2,
            'IN': 1,
            '.': 1,
        }

        self.vocabulary = Vocabulary(
                counter={
                    'words': words, 
                    'pos_tag': tags, 
                    'adv_tag': xpos,
                    'label': labels,
                    }
            )
        
        self.data_sentence1 = ['this', 'buzzmachine', 'post', 'argues', 'that', 'google', "'s", 'rush', 'toward',
             'ubiquity', 'might', 'backfire', '--', 'which', 'we', "'ve", 'all', 'heard', 'before', ',', 'but', 
             'it', "'s", 'particularly', 'well', '-', 'put', 'in', 'this', 'post', '.']
        self.data_word_tags1 = ['DET', 'PROPN', 'NOUN', 'VERB', 'SCONJ', 'PROPN', 'PART', 'NOUN', 'ADP', 'NOUN', 'AUX', 'VERB', 'PUNCT',
             'PRON', 'PRON', 'AUX', 'ADV', 'VERB', 'ADV', 'PUNCT', 'CCONJ', 'PRON', 'AUX', 'ADV', 'ADV', 'PUNCT', 'VERB', 'ADP', 'DET', 
             'NOUN', 'PUNCT']
        self.data_word_xpos1 = ['DT', 'NNP', 'NN', 'VBZ', 'IN', 'NNP', 'POS', 'NN', 'IN', 'NN', 'MD', 'VB', ',', 
             'WDT', 'PRP', 'VBP', 'RB', 'VBN', 'RB', ',', 'CC', 'PRP', 'VBZ', 'RB', 'RB', 'HYPH', 'VBN', 'IN',
              'DT', 'NN', '.']
        self.data_word_labs1 = ['det', 'compound', 'nsubj', 'root', 'mark', 'nmod:poss', 'case', 'nsubj', 'case', 'nmod', 'aux', 'ccomp',
             'punct', 'obj', 'nsubj', 'aux', 'advmod', 'acl:relcl', 'advmod', 'punct', 'cc', 'nsubj:pass', 'cop', 'advmod', 'advmod', 'punct',
             'conj', 'case', 'det', 'obl', 'punct']
        self.correct_parents1 = [3, 3, 4, 0, 12, 8, 6, 12, 10, 8, 12, 4, 12, 18, 18, 18, 18, 12, 18, 27, 27, 27, 27, 27, 27, 27, 4, 30, 30, 27, 4]
        
        self.data_sentence2 = ['i', "'m", 'staying', 'away', 'from', 'the', 'stock', '.']
        self.data_word_xpos2 = ['PRP', 'VBP', 'VBG', 'RB', 'IN', 'DT', 'NN', '.']
        self.data_word_tags2 = ['PRON', 'AUX', 'VERB', 'ADV', 'ADP', 'DET', 'NOUN', 'PUNCT']
        self.data_word_labs2 = ['nsubj', 'aux', 'root', 'advmod', 'case', 'det', 'obl', 'punct']
        self.correct_parents2 = [3, 3, 0, 3, 7, 7, 4, 3]

        self.sentence = ['this', 'is', 'a', 'test', 'for', 'this', 'vocabulary', '.']
        self.word_xpos = ['DT', 'VBZ', 'DT', 'NN', 'IN', 'DT', 'NN', '.']       
        self.word_tags = ['PRON', 'VERB', 'DET', 'NOUN', 'ADP', 'PRON', 'NOUN', 'PUNCT']
        self.word_labs = ['nsubj','conj', 'det', 'compound', 'case', 'nsubj', 'obj', 'punct']

    def test_sentence_dataset_with_vocabulary(self):
        dataset = SentenceDataset(self.file, ROOT_TOKEN, ROOT_TAG, ROOT_LABEL, vocab=self.vocabulary)
        self.assertEqual(len(dataset),1, 'Expected length to be of 1 sentence.')
        
        index, tags, xpos, parents, labels, word_tags = dataset[0]
        
        root_idx = self.vocabulary.get_token_index(ROOT_TOKEN, 'words')
        tag_idx = self.vocabulary.get_token_index(ROOT_TAG, 'pos_tag')
        xpos_idx = self.vocabulary.get_token_index(ROOT_TAG, 'adv_tag')
        label_idx = self.vocabulary.get_token_index(ROOT_LABEL, 'label')
        
        correct_idxs = [root_idx] + [self.vocabulary.get_token_index(w, 'words') for w in self.sentence]
        correct_tags = [tag_idx] + [self.vocabulary.get_token_index(w, 'pos_tag') for w in self.word_tags]
        correct_xpos = [xpos_idx] + [self.vocabulary.get_token_index(w, 'adv_tag') for w in self.word_xpos]
        correct_labs = [label_idx] + [self.vocabulary.get_token_index(w, 'label') for w in self.word_labs]
        correct_word_tags = [ROOT_TAG] + self.word_xpos
        correct_parents = [0] + list(range(len(self.sentence)))


        self.assertEqual(index, correct_idxs, 'Incorrect indexing for sentence')
        self.assertEqual(tags, correct_tags, 'Incorrect tags for sentence')
        self.assertEqual(xpos, correct_xpos, 'Incorrect advanced pos tags for sentence')
        self.assertEqual(parents, correct_parents, 'Incorrect parents for sentence')
        self.assertEqual(labels, correct_labs, 'Incorrect labels for sentence')
        self.assertEqual(word_tags, correct_word_tags, 'Incorrect word tags')

    def test_sentence_dataset_no_vocabulary(self):
        dataset = SentenceDataset(self.data_file, ROOT_TOKEN, ROOT_TAG, ROOT_LABEL)
        self.assertEqual(len(dataset),2, 'Expected length to be of 2 sentences.')
        
        vocabulary = dataset.getVocabulary()

        self.assertEqual(vocabulary.get_token_index('worddoesntexist', 'words'), 1, 'Missing words should have index 1')
        index, tags, xpos, parents, labels, word_tags = dataset[0]

        root_idx = vocabulary.get_token_index(ROOT_TOKEN, 'words')
        tag_idx = vocabulary.get_token_index(ROOT_TAG, 'pos_tag')
        xpos_idx = vocabulary.get_token_index(ROOT_TAG, 'adv_tag')
        label_idx = vocabulary.get_token_index(ROOT_LABEL, 'label')
                
        correct_idxs = [root_idx] + [vocabulary.get_token_index(w, 'words') for w in self.data_sentence1]
        correct_tags = [tag_idx] + [vocabulary.get_token_index(w, 'pos_tag') for w in self.data_word_tags1]
        correct_xpos = [xpos_idx] + [vocabulary.get_token_index(w, 'adv_tag') for w in self.data_word_xpos1]
        correct_labs = [label_idx] + [vocabulary.get_token_index(w, 'label') for w in self.data_word_labs1]
        correct_word_tags = [ROOT_TAG] + self.data_word_xpos1
        correct_parents = [0] + self.correct_parents1


        self.assertEqual(index, correct_idxs, 'Incorrect indexing for sentence')
        self.assertEqual(tags, correct_tags, 'Incorrect tags for sentence')
        self.assertEqual(xpos, correct_xpos, 'Incorrect advanced pos tags for sentence')
        self.assertEqual(parents, correct_parents, 'Incorrect parents for sentence')
        self.assertEqual(labels, correct_labs, 'Incorrect labels for sentence')
        self.assertEqual(word_tags, correct_word_tags, 'Incorrect word tags')

        index, tags, xpos, parents, labels, word_tags = dataset[1]

        correct_idxs = [root_idx] + [vocabulary.get_token_index(w, 'words') for w in self.data_sentence2]
        correct_tags = [tag_idx] + [vocabulary.get_token_index(w, 'pos_tag') for w in self.data_word_tags2]
        correct_xpos = [xpos_idx] + [vocabulary.get_token_index(w, 'adv_tag') for w in self.data_word_xpos2]
        correct_labs = [label_idx] + [vocabulary.get_token_index(w, 'label') for w in self.data_word_labs2]
        correct_word_tags = [ROOT_TAG] + self.data_word_xpos2
        correct_parents = [0] + self.correct_parents2

        self.assertEqual(index, correct_idxs, 'Incorrect indexing for sentence')
        self.assertEqual(tags, correct_tags, 'Incorrect tags for sentence')
        self.assertEqual(xpos, correct_xpos, 'Incorrect advanced pos tags for sentence')
        self.assertEqual(parents, correct_parents, 'Incorrect parents for sentence')
        self.assertEqual(labels, correct_labs, 'Incorrect labels for sentence')
        self.assertEqual(word_tags, correct_word_tags, 'Incorrect word tags')
    
    @unittest.expectedFailure
    def test_embedding_dataset_dimension(self):
        words_to_index = self.vocabulary.get_token_to_index_vocabulary(namespace='words')
        dataset = EmbeddingDataset(self.embedding_file, 50, words_to_index, ROOT_TOKEN)
        self.assertEqual(dataset[0], np.zeros(7, dtype=float))

    def test_embedding_dataset_values(self):
        words_to_index = self.vocabulary.get_token_to_index_vocabulary(namespace='words')
        dataset = EmbeddingDataset(self.embedding_file, 7, words_to_index, ROOT_TOKEN)
        
        self.assertEqual(len(dataset), 10) # 7 words + (padding, unknown and root)

        for idx in range(1, 4):
            tidx = words_to_index[self.sentence[idx-1]]
            embed = np.zeros(7, dtype=float)
            embed[idx-1] = float(idx)
            self.assertTrue(np.all(embed == dataset[tidx]))

        self.assertTrue(np.all(np.zeros(7, dtype=float) == dataset[words_to_index['.']]), 'Embeddings do not correspond')
        self.assertTrue(np.all(np.zeros(7, dtype=float) == dataset[self.vocabulary.get_token_index('wordnothere', 'words')]), 
            'The embedding of a word that does not belong in the vocabulary should be 0')

class PadderTestCase(unittest.TestCase):
    def test_labelled_padder_all_equal(self):
        sample1 = ([1, 2], [3, 4], [5, 6], [7, 8], [9, 10], ['tag1', 'tag2'])
        sample2 = ([13, 14], [15, 16], [17, 18], [19, 20], [21, 22], ['tag1', 'tag2'])
        padded = labelled_padder([sample1, sample2])

        sentences = padded['sentence']
        tags = padded['tags']
        xpos = padded['xpos']
        parents = padded['parents']
        labels = padded['labels']
        lengths = padded['lengths']
        word_xpos = padded['word_tags']

        self.assertEqual(sentences.shape, tags.shape)
        self.assertEqual(sentences.shape, xpos.shape)
        self.assertEqual(sentences.shape, parents.shape)
        self.assertEqual(sentences.shape, labels.shape)
        self.assertEqual(len(lengths), 2)
        self.assertEqual(len(word_xpos), 2)
        self.assertEqual(len(word_xpos[0]), len(word_xpos[1]))
        self.assertTrue(torch.all(parents[:, 0] == -1))
    
    def test_labelled_padder_not_equal(self):
        sample1 = ([1, 2], [3, 4], [5, 6], [7, 8], [9, 10], ['tag1', 'tag2'])
        sample2 = ([13, 14, 1], [15, 16, 1], [17, 18, 1], [19, 20, 1], [21, 22, 1], ['tag1', 'tag2', 'tag3'])

        padded = labelled_padder([sample1, sample2])

        sentences = padded['sentence']
        tags = padded['tags']
        xpos = padded['xpos']
        parents = padded['parents']
        labels = padded['labels']
        lengths = padded['lengths']
        word_xpos = padded['word_tags']

        self.assertEqual(sentences.shape, (2,3))
        self.assertEqual(sentences.shape, tags.shape)
        self.assertEqual(sentences.shape, xpos.shape)
        self.assertEqual(sentences.shape, parents.shape)
        self.assertEqual(sentences.shape, labels.shape)
        self.assertEqual(len(lengths), 2)
        self.assertEqual(len(word_xpos), 2)
        self.assertTrue(torch.all(parents[:, 0] == -1))

        self.assertEqual(sentences[0][2], 0)
        self.assertEqual(tags[0][2], 0)
        self.assertEqual(xpos[0][2], 0)
        self.assertEqual(parents[0][2], -1)
        self.assertEqual(labels[0][2], 0)
    
    def test_unlabelled_padder_all_equal(self):
        features = np.random.rand(40, 40, 5)
        array1 =[1] * 40
        array2 = [2] * 40
        array3 = [3] * 40

        sample1 = (array1, array1, array1, array1, array1, ['tag'] * 40, features)
        sample2 = (array2, array2, array2, array2, array2, ['tag'] * 40, features)
        sample3 = (array3, array3, array3, array3, array3, ['tag'] * 40, features)
        
        padded = unlabelled_padder([sample1, sample2, sample3])

        sentences = padded['sentence']
        tags = padded['tags']
        xpos = padded['xpos']
        parents = padded['parents']
        lengths = padded['lengths']
        labels = padded['labels']
        word_xpos = padded['word_tags']

        self.assertEqual(sentences.shape, (3,40))
        self.assertEqual(sentences.shape, tags.shape)
        self.assertEqual(sentences.shape, xpos.shape)
        self.assertEqual(sentences.shape, parents.shape)
        self.assertEqual(sentences.shape, labels.shape)
        self.assertEqual(len(lengths), 3)
        self.assertEqual(len(word_xpos), 3)
        self.assertEqual(len(word_xpos[0]), len(word_xpos[1]))
        self.assertEqual(len(word_xpos[0]), len(word_xpos[1]))
        self.assertTrue(torch.all(parents[:, 0] == -1))

    def test_unlabelled_padder_not_equal(self):
        features1 = np.random.rand(2,2,11)
        features2 = np.random.rand(10,10,11)
        array1 = [1] * 2
        array2 = [1] * 10
        sample1 = (array1, array1, array1, array1, array1, ['tag1', 'tag2'], features1)
        sample2 = (array2, array2, array2, array2, array2, ['tag1'] * 10, features2)

        padded = unlabelled_padder([sample1, sample2])
        
        sentences = padded['sentence']
        tags = padded['tags']
        xpos = padded['xpos']
        parents = padded['parents']
        labels = padded['labels']
        features = padded['features']
        lengths = padded['lengths']
        word_xpos = padded['word_tags']

        self.assertEqual(sentences.shape, (2,10))
        self.assertEqual(sentences.shape, tags.shape)
        self.assertEqual(sentences.shape, xpos.shape)
        self.assertEqual(sentences.shape, parents.shape)
        self.assertEqual(sentences.shape, labels.shape)
        self.assertEqual(len(lengths), 2)
        self.assertEqual(len(word_xpos), 2)
        self.assertTrue(torch.all(parents[:, 0] == -1))

        self.assertTrue(torch.all(sentences[0][2:] == 0))
        self.assertTrue(torch.all(tags[0][2:] == 0))
        self.assertTrue(torch.all(xpos[0][2:] == 0))
        self.assertTrue(torch.all(parents[0][2:] == -1))
        self.assertTrue(torch.all(labels[0][2:] == 0))

        self.assertEqual(features.shape, (2,10,10,11))
        self.assertTrue(torch.all(features[0, 2:, :] == torch.zeros(11)))
        self.assertTrue(torch.all(features[0, :, 2:] == torch.zeros(11)))





    

        
