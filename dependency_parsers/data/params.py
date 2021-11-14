UD_ENGLISH_EWT_TRAIN = 'en_ewt-ud-train.conllu'
UD_ENGLISH_EWT_DEV  = 'en_ewt-ud-dev.conllu'
UD_ENGLISH_EWT_TEST = 'en_ewt-ud-test.conllu'

EMBEDDING_FILE = 'glove.6B.50d.txt'
OBJECT_FILE = 'dependency_parsers/data/cache.pickle'
PARAM_FILE = 'dependency_parsers/data/parameters.pickle'

ROOT_TOKEN = '@@ROOT@@'
ROOT_TAG   = '@@ROOT@@'
ROOT_LABEL = '@@root@@' # no incoming edge

ROOT_IDX = 0
PAD_IDX = 0

PAD_VALUE = -100

FILE_SIZE = 20000
TRAIN_SPLIT = 1600
VAL_SPLIT = 400

EMBEDDING_DIM = 50
BATCH_SIZE = 32
HIDDEN_DIM = 128
ARC_DIM = 128
LAB_DIM = 128
NUM_LAYERS = 2
DROPOUT = 0.1
NUM_EPOCH = 20