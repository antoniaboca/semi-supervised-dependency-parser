import pickle
from processor import SentenceDataset, EmbeddingDataset, Embed
from params import UD_ENGLISH_EWT_DEV, FILE_SIZE, EMBEDDING_FILE, EMBEDDING_DIM, OBJECT_FILE, PARAM_FILE

print('Load the sentences from {}...'.format(UD_ENGLISH_EWT_DEV))
training_set = SentenceDataset(UD_ENGLISH_EWT_DEV, max_size=FILE_SIZE)
print('Loaded the training set.')

print('Load the embeddings from {}...'.format(EMBEDDING_FILE))
embeddings = EmbeddingDataset(EMBEDDING_FILE, EMBEDDING_DIM, training_set.word_to_index) 
print('Loaded the embeddings.')

training_set.transform = Embed(embeddings)
embedded_set = [training_set[idx] for idx in range(len(training_set))] # now this set is a list of (sentence, embedding, tags)

print('Save the curated set to a file using pickle...')
with open(OBJECT_FILE, 'wb') as file:
    pickle.dump(embedded_set, file)
print('Saved.')

print('Save other size parameters...')
with open(PARAM_FILE, 'wb') as file:
    pickle.dump({'TAGSET_SIZE': len(training_set.pos_to_index)}, file)
print('Saved.')
