import pickle
from .processor import SentenceDataset, EmbeddingDataset, Embed
from .params import OBJECT_FILE, PARAM_FILE
from .params import ROOT_TOKEN, ROOT_LABEL, ROOT_TAG

def data_load(args):
    TRAIN_FILE = args.train_data
    VAL_FILE = args.validation_data
    TEST_FILE = args.testing_data
    EMBEDDING_FILE = args.embeddings
    EMBEDDING_DIM = args.embedding_dim
    SENTENCE_SIZE = args.limit_sentence_size
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

    print('Save the curated set to a file using pickle...')
    with open(OBJECT_FILE, 'wb') as file:
        pickle.dump({'train': embedded_set, 'test': embed_test_set, 'dev': embed_dev_set, 'embeddings': embeddings}, file)
    print('Saved.')

    print('Save other size parameters...')
    with open(PARAM_FILE, 'wb') as file:
        pickle.dump({
            'TAGSET_SIZE': len(training_set.pos_to_index), 
            'LABSET_SIZE': len(training_set.label_to_index)
            }, file)
    print('Saved.')

