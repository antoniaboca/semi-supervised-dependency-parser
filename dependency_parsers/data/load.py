import pickle
import random as rand
import copy 
import math

from collections import deque
from .processor import SentenceDataset, EmbeddingDataset, Embed
# from .params import OBJECT_FILE
from .params import OBJECT_FILE, ROOT_TOKEN, ROOT_LABEL, ROOT_TAG

def file_load(args):
    TRAIN_FILE = args.train_data
    VAL_FILE = args.validation_data
    TEST_FILE = args.testing_data
    EMBEDDING_FILE = args.embeddings
    EMBEDDING_DIM = args.embedding_dim
    SENTENCE_SIZE = args.limit_sentence_size
    OBJECT_FILE = args.save_to_pickle_file

    if SENTENCE_SIZE == 0:
        SENTENCE_SIZE = None
    
    print('Load the sentences from {}...'.format(TRAIN_FILE))
    training_set = SentenceDataset(TRAIN_FILE, ROOT_TOKEN, ROOT_TAG, ROOT_LABEL, sentence_len=SENTENCE_SIZE)
    print('Loaded the training set. Number of sentences: {}'.format(len(training_set.sentences)))

    print('Load the embeddings from {}...'.format(EMBEDDING_FILE))
    embeddings = EmbeddingDataset(EMBEDDING_FILE, EMBEDDING_DIM, training_set.word_to_index, ROOT_TOKEN).idx_embeds
    print('Loaded the embeddings.')

    print('Load the test set from {}...'.format(TEST_FILE))
    test_set = SentenceDataset(TEST_FILE, ROOT_TOKEN, ROOT_TAG, ROOT_LABEL, vocab=training_set.vocabulary, sentence_len=SENTENCE_SIZE)
    print('Loaded the test set. Number of sentences: {}'.format(len(test_set.sentences)))

    print('Load the dev set from {}...'.format(VAL_FILE))
    dev_set = SentenceDataset(VAL_FILE, ROOT_TOKEN, ROOT_TAG, ROOT_LABEL,vocab=training_set.vocabulary, sentence_len=SENTENCE_SIZE)
    print('Loaded the dev set. Number of sentences: {}'.format(len(dev_set.sentences)))

    # training_set.transform = Embed(embeddings)
    embedded_set = [training_set[idx] for idx in range(len(training_set))] # now this set is a list of (sentence, tags, parent, label)

    # test_set.transform = Embed(embeddings)
    embed_test_set = [test_set[idx] for idx in range(len(test_set))]

    # dev_set.transform = Embed(embeddings)
    embed_dev_set = [dev_set[idx] for idx in range(len(dev_set))]

    vocab = training_set.vocabulary
    return embedded_set, embed_test_set, embed_dev_set, embeddings, vocab, len(training_set.pos_to_index), len(training_set.label_to_index)

def file_save(args):
    train_set, test_set, dev_set, embeddings, vocab, tag_size, label_size = file_load(args)
    OBJECT_FILE = args.save_to_pickle_file
    
    print('Save the curated set to a file using pickle...')
    with open(OBJECT_FILE, 'wb') as file:
        pickle.dump({
            'train_labelled': train_set, 
            'test': test_set, 
            'dev': dev_set, 
            'embeddings': embeddings,
            'vocabulary': vocab,
            'TAGSET_SIZE': tag_size, 
            'LABSET_SIZE': label_size
        }, file)
    print('Saved.')

def create_buckets(set):
    buckets = {}
    for sentence, tag, xpos, parent, label, word_tag in set:
        val = len(sentence)
        if val not in buckets.keys():
            buckets[val] = []
        buckets[val].append((sentence, tag, xpos, parent, label, word_tag))
    return buckets

def stratified_random_sampling(train_buckets, train_set, size):
    rand_set = []
    remainder = []
    sampled = 0
    for slen, elems in train_buckets.items():
        sample_size = math.floor(1.0 * (len(elems) / len(train_set)) * size)
        sampled += sample_size

        sample = {*rand.sample(range(len(elems)), sample_size)}
        for idx in range(len(elems)):
            if idx in sample:
                rand_set.append(elems[idx])
            else:
                remainder.append(elems[idx])
    
    rand.shuffle(remainder)
    rand_set.extend(remainder[:(size - sampled)])
    remainder = remainder[(size - sampled):]
    rand.shuffle(rand_set)

    return rand_set, remainder

def bucket_save(train_buckets, loaded, file_name, size):
    train_set, test_set, dev_set, embeddings, vocab, tag_size, label_size = loaded
    rand_set, remainder = stratified_random_sampling(train_buckets, train_set, size)

    print(f'Saving sample of size {len(train_set)} to {file_name}...')
    with open(file_name, 'wb') as file:
        pickle.dump({'train_labelled': rand_set, 
            'test': test_set, 
            'dev': dev_set, 
            'embeddings': embeddings, 
            'vocabulary': vocab,
            'TAGSET_SIZE': tag_size, 
            'LABSET_SIZE': label_size,
            'remainder': remainder,
            }, file)
    print('Saved.')

def bucket_unlabelled_save(train_buckets, loaded, file_name, size):
    train_set, test_set, dev_set, embeddings, vocab, tag_size, label_size = loaded
    labelled_set, unlabelled_set = stratified_random_sampling(train_buckets, train_set, size)

    for _, tags, _, _, _, _ in unlabelled_set:
        for tag in tags:
            assert tag < 20

    print(f'Saving labelled sample of size {size} AND the rest of unlabelled samples to {file_name}...')
    with open(file_name, 'wb') as file:
        pickle.dump({'train_labelled': labelled_set, 
            'train_unlabelled': unlabelled_set,
            'test': test_set, 
            'dev': dev_set, 
            'embeddings': embeddings, 
            'vocabulary': vocab,
            'TAGSET_SIZE': tag_size, 
            'LABSET_SIZE': label_size
            }, file)
    print('Saved.')

def bucket_loop(args):
    loaded = file_load(args)
    train_buckets = create_buckets(loaded[0])
    data_size = [1000, 2000, 4000, 6000, 8000, 10000, 12000]
    for size in data_size:
        aux_buckets = copy.deepcopy(train_buckets)
        bucket_save(aux_buckets, loaded, 'train' + str(size) + '.pickle', size)

def bucket_unlabelled_loop(args):
    loaded = file_load(args)
    train_buckets = create_buckets(loaded[0])
    data_size = [100, 500, 1000, 4000]
    for size in data_size:
        aux_buckets = copy.deepcopy(train_buckets)
        bucket_unlabelled_save(aux_buckets, loaded, 'unlabelled' + str(size) + '.pickle', size)