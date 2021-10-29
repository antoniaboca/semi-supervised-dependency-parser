import pickle
from processor import SentenceDataset, EmbeddingDataset, Embed
from params import UD_ENGLISH_EWT_TRAIN, UD_ENGLISH_EWT_TEST, UD_ENGLISH_EWT_DEV, FILE_SIZE, EMBEDDING_FILE, EMBEDDING_DIM, OBJECT_FILE, PARAM_FILE

print('Load the sentences from {}...'.format(UD_ENGLISH_EWT_TRAIN))
training_set = SentenceDataset(UD_ENGLISH_EWT_TRAIN, max_size=FILE_SIZE)
print('Loaded the training set. Number of sentences: {}'.format(len(training_set.sentences)))

print('Load the embeddings from {}...'.format(EMBEDDING_FILE))
embeddings = EmbeddingDataset(EMBEDDING_FILE, EMBEDDING_DIM, training_set.word_to_index) 
print('Loaded the embeddings.')

print('Load the test set from {}...'.format(UD_ENGLISH_EWT_TEST))
test_set = SentenceDataset(UD_ENGLISH_EWT_TEST, vocab=training_set.vocabulary)
print('Loaded the test set. Number of sentences: {}'.format(len(test_set.sentences)))

print('Load the dev set from {}...'.format(UD_ENGLISH_EWT_DEV))
dev_set = SentenceDataset(UD_ENGLISH_EWT_DEV, vocab=training_set.vocabulary)
print('Loaded the dev set. Number of sentences: {}'.format(len(dev_set.sentences)))

training_set.transform = Embed(embeddings)
embedded_set = [training_set[idx] for idx in range(len(training_set))] # now this set is a list of (sentence, embedding, tags, parent)

test_set.transform = Embed(embeddings)
embed_test_set = [test_set[idx] for idx in range(len(test_set))]

dev_set.transform = Embed(embeddings)
embed_dev_set = [dev_set[idx] for idx in range(len(dev_set))]

print('Save the curated set to a file using pickle...')
with open(OBJECT_FILE, 'wb') as file:
    pickle.dump({'train': embedded_set, 'test': embed_test_set, 'dev': embed_dev_set}, file)
print('Saved.')

print('Save other size parameters...')
with open(PARAM_FILE, 'wb') as file:
    pickle.dump({'TAGSET_SIZE': len(training_set.pos_to_index)}, file)
print('Saved.')

