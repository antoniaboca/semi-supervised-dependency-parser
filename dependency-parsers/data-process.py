import pyconll 
import pyconll.util

UD_ENGLISH_GUM = 'dependency-parsers/en_gum-ud-train.conllu'

data_load = pyconll.load_from_file(UD_ENGLISH_GUM)

first_sentence = data_load[0]
for token in first_sentence:
    print(token.lemma)

