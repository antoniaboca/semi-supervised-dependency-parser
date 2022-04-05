import pickle
import random as rand
import copy 
import math

from collections import deque
from .processor import SentenceDataset, EmbeddingDataset, Embed
# from .params import OBJECT_FILE
from .params import OBJECT_FILE, ROOT_TOKEN, ROOT_LABEL, ROOT_TAG

def file_load(args):
    """
    Function to load the training, validation and testing datasets and 
    the Embeddings file.

    Args:
        args (object): Object containg command line arguments used to 
            configure loading and processing parameters.

    Returns:
        tuple: Tuple of PyTorch Dataset objects: (list of training sentences, 
            list of test sentences, list of validation sentences, embeddings array, 
            the number of UPOS tags, the number of edge labels found)
    """
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
    """Function that saves datasets to pickle files.

    Args:
        args (object): Object containg command line arguments used to 
            configure loading and processing parameters.
    """
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
    """Function that creates buckets of sentences based on their length.

    Args:
        set (list): List of sentences to bucket.

    Returns:
        dict: Dictionary where each key is a sentence length and each value
            is a list of tuples; each tuple contains the indexed sentence and 
            its annotations.
    """
    buckets = {}
    for sentence, tag, xpos, parent, label, word_tag in set:
        val = len(sentence)
        if val not in buckets.keys():
            buckets[val] = []
        buckets[val].append((sentence, tag, xpos, parent, label, word_tag))
    return buckets

def stratified_random_sampling(train_buckets, train_set, size):
    """Algorithm to do stratified random sampling from buckets of lengths.

    Args:
        train_buckets (dict): Dictionary where each key is a sentence length and each value
            is a list of tuples; each tuple contains the indexed sentence and 
            its annotations.
        train_set (list): Unbucketed sentence set.
        size (int): Size of the random sample.

    Returns:
        tuple: Pair of sampled set of sentences and the remaining sentences.
    """
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
    """
    Function that creates a sample from a dictionary of buckets and saves the sample
    to a pickle file (for the supervised context).

    Args:
        train_buckets (dict): Dictionary where each key is a sentence length and each value
            is a list of tuples; each tuple contains the indexed sentence and 
            its annotations.
        loaded (tuple): Tuple of PyTorch Dataset objects: (list of training sentences, 
            list of test sentences, list of validation sentences, embeddings array, 
            the number of UPOS tags, the number of edge labels found)
        file_name (str): File to save the sample to.
        size (int): Size of desired sample.
    """
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
    """ 
    Function that creates a sample from a dictionary of buckets and saves the sample
    to a pickle file (for the semi-supervised context). This function splits the training
    set into 'labelled' and 'unlabelled' sentences. The labelled set is sampled using 
    stratified random sampling, while the unlabelled set is formed of the remaining
    sentences.

    Args:
        train_buckets (dict): Dictionary where each key is a sentence length and each value
            is a list of tuples; each tuple contains the indexed sentence and 
            its annotations.
        loaded (tuple): Tuple of PyTorch Dataset objects: (list of training sentences, 
            list of test sentences, list of validation sentences, embeddings array, 
            the number of UPOS tags, the number of edge labels found)
        file_name (str): File to save the sample to.
        size (int): Size of the labelled sample.
    """
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
    """
    Function that creates samples of labelled data for the supervised context
    for 7 different sample sizes.

    Args:
        args (object): Object containg command line arguments used to 
            configure loading and processing parameters.
    """
    loaded = file_load(args)
    train_buckets = create_buckets(loaded[0])
    data_size = [1000, 2000, 4000, 6000, 8000, 10000, 12000]
    for size in data_size:
        aux_buckets = copy.deepcopy(train_buckets)
        bucket_save(aux_buckets, loaded, 'train' + str(size) + '.pickle', size)

def bucket_unlabelled_loop(args):
    """Function that creates samples of labelled and unlabelled data for the
    semi-supervised context for 4 different sample sizes.

    Args:
        args (object): Object containg command line arguments used to 
            configure loading and processing parameters.
    """
    loaded = file_load(args)
    train_buckets = create_buckets(loaded[0])
    data_size = [100, 500, 1000, 4000]
    for size in data_size:
        aux_buckets = copy.deepcopy(train_buckets)
        bucket_unlabelled_save(aux_buckets, loaded, 'unlabelled' + str(size) + '.pickle', size)