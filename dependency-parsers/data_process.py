import pyconll 
import pyconll.util
from allennlp.data.vocabulary import Vocabulary

UD_ENGLISH_GUM = 'dependency-parsers/en_gum-ud-train.conllu'

class DataProcessor:
    def __init__(self, file, max_size=None):
        self.file = file
        self.words = {}
        self.pos_tags = {}
        self.dep_rel = {}
        self.index_to_token_tuples = []

        self.data = pyconll.load_from_file(file)
        self.sentences = []

        count = 0
        for sentence in self.data:
            word_list = []
            tag_list = []

            count += 1
            if count > max_size:
                break

            for token in sentence:
                if token.lemma is None:
                    continue
                if token.lemma in self.words:
                    self.words[token.lemma] += 1
                else:
                    self.words[token.lemma] = 1
                word_list.append(token.lemma)

                if token.upos is None:
                    continue

                if token.upos in self.pos_tags:
                    self.pos_tags[token.upos] += 1
                else:
                    self.pos_tags[token.upos] = 1
                tag_list.append(token.upos)

                if token.deprel is None:
                    continue
                if token.deprel in self.dep_rel:
                    self.dep_rel[token.deprel] += 1
                else:
                    self.dep_rel[token.deprel] = 1
            
            self.sentences.append((word_list, tag_list))


        self.vocab = Vocabulary(counter={'words': self.words, 'pos_tags': self.pos_tags, 'dep_rel': self.dep_rel})
        self.index_to_token = self.vocab.get_index_to_token_vocabulary(namespace='words')
        self.index_to_pos = self.vocab.get_index_to_token_vocabulary(namespace='pos_tags')
        
        self.index_to_token_tuples = [(key, value) for key, value in self.index_to_token.items()]
        self.index_to_token_tuples.sort()
        
        self.word_to_index = self.vocab.get_token_to_index_vocabulary(namespace='words')
        self.pos_to_index = self.vocab.get_token_to_index_vocabulary(namespace='pos_tags')