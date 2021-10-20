import numpy as np
import torch

class GloveEmbedding:
    def __init__(self, filename, dim, words_to_index):
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

        indexed = np.zeros((len(words_to_index), self.dim), dtype=float)
        for word, values in self.embeddings.items():
            if word in words_to_index:
                indexed[words_to_index[word]] = values
        
        self.idx_embeds = torch.tensor(indexed)
    



