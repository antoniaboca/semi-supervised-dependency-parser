import pyconll 
import pyconll.util
from collections import OrderedDict

UD_ENGLISH_GUM = 'dependency-parsers/en_gum-ud-train.conllu'

data_load = pyconll.load_from_file(UD_ENGLISH_GUM)

counter = {}
for sentence in data_load:
    for token in sentence:
        if token.lemma in counter:
            counter[token.lemma] += 1
        else:
            counter[token.lemma] = 1

vocab = sorted([(k, v) for k, v in counter.items()], key=lambda t:t[1])
vocab.reverse()

vocab_dict = OrderedDict(vocab)

for k, v in vocab_dict.items():
    print("{}: {}".format(k,v))

