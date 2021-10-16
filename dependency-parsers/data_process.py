import pyconll 
import pyconll.util
from allennlp.data.vocabulary import Vocabulary

UD_ENGLISH_GUM = 'dependency-parsers/en_gum-ud-train.conllu'

class DataProcessor:
    def __init__(self, file):
        self.file = file
        self.words = {}
        self.pos_tags = {}
        self.dep_rel = {}

        data_load = pyconll.load_from_file(file)

        for sentence in data_load:
            for token in sentence:
                if token.lemma is None:
                    continue
                if token.lemma in self.words:
                    self.words[token.lemma] += 1
                else:
                    self.words[token.lemma] = 1

                if token.upos is None:
                    continue

                if token.upos in self.pos_tags:
                    self.pos_tags[token.upos] += 1
                else:
                    self.pos_tags[token.upos] = 1

                if token.deprel is None:
                    continue
                if token.deprel in self.dep_rel:
                    self.dep_rel[token.deprel] += 1
                else:
                    self.dep_rel[token.deprel] = 1


        self.vocab = Vocabulary(counter={'words': self.words, 'pos_tags': self.pos_tags, 'dep_rel': self.dep_rel})
