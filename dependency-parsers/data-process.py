import pyconll 
import pyconll.util
from collections import OrderedDict
from allennlp.data.vocabulary import Vocabulary

UD_ENGLISH_GUM = 'dependency-parsers/en_gum-ud-train.conllu'

data_load = pyconll.load_from_file(UD_ENGLISH_GUM)

words = {}
pos_tags = {}
dep_rel = {}

for sentence in data_load:
    for token in sentence:
        if token.lemma is None:
            continue
        if token.lemma in words:
            words[token.lemma] += 1
        else:
            words[token.lemma] = 1

        if token.upos is None:
            continue

        if token.upos in pos_tags:
            pos_tags[token.upos] += 1
        else:
            pos_tags[token.upos] = 1

        if token.deprel is None:
            continue
        if token.deprel in dep_rel:
            dep_rel[token.deprel] += 1
        else:
            dep_rel[token.deprel] = 1

vocab = Vocabulary(counter={'words': words, 'pos_tags': pos_tags, 'dep_rel': dep_rel})

print(vocab)