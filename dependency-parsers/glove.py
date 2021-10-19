import numpy as np
import torch

class GloveEmbedding:
    def __init__(self, filename, dim):
        self.embeddings = {}
        self.dim = dim

        with open(filename) as file:
            for line in file:
                tokens = line.split()
                word = tokens[0]
                embeds = []
                for idx in range(1, dim + 1):
                    embeds.append(float(tokens[idx]))
                
                self.embeddings[word] = np.asarray(embeds)

        print(len(self.embeddings))
    def get_pretrained_from_index(self, words_to_index):
        #import ipdb; ipdb.set_trace()
        indexed = np.zeros((len(words_to_index), self.dim), dtype=float)
        for word, values in self.embeddings.items():
            if word in words_to_index:
                indexed[words_to_index[word]] = values
        
        idx_embeds = torch.tensor(indexed)
        return idx_embeds
    



