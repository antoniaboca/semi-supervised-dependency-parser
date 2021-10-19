from struct import pack
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import loss
import torch.optim as optim

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from torch.nn import Embedding, Linear, LSTM
from data_process import DataProcessor

UD_ENGLISH_GUM = 'dependency-parsers/en_gum-ud-train.conllu'
EMBEDDING_DIM = 10
HIDDEN_DIM = 32
NUM_LAYERS = 2
DROPOUT = 0.1
FILE_SIZE = 1000
NUM_EPOCH = 10
BATCH_SIZE = 10

def index_sequence(seq, to_index):
    idxs = [to_index[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)

def batch_gen(training_set, tag_set, BATCH):
    idx = 0
    while idx < len(training_set):
        sentences = [training_set[idx + offset]
            for offset in range(min(BATCH, len(training_set) - idx))] 

        sent_lens = [len(training_set[idx + offset]) 
            for offset in range(min(BATCH, len(training_set) - idx))]

        tags = [tag_set[idx + offset]
                for offset in range(min(BATCH, len(training_set) - idx))]
        
        padded_sentences = pad_sequence(sentences, batch_first=True, padding_value=0)
        padded_tags = pad_sequence(tags, batch_first=True, padding_value=0)

        yield (padded_sentences, torch.tensor(sent_lens), padded_tags)

        idx += 10
        

class LSTMTagger(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_layers, dropout, vocab_size, tagset_size):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = Embedding(vocab_size, embedding_dim, padding_idx=0)

        self.lstm = LSTM(
            input_size=embedding_dim, 
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=True
        )

        self.hidden2tag = Linear(hidden_dim * 2, tagset_size)
    
    def forward(self, tensor_batch, sent_lens):
        embeds = self.word_embeddings(tensor_batch)

        lstm_input = pack_padded_sequence(embeds, sent_lens, batch_first=True, enforce_sorted=False)

        lstm_out, _ = self.lstm(lstm_input)
        lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)
        
        tag_space = self.hidden2tag(lstm_out)
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores


processor = DataProcessor(UD_ENGLISH_GUM, FILE_SIZE)

words_to_index = processor.word_to_index
pos_to_index = processor.pos_to_index
training_data = processor.sentences
training_indexed = []
tags_indexed = []

for sentence, tags in training_data:
    training_indexed.append(index_sequence(sentence, words_to_index))
    tags_indexed.append(index_sequence(tags, pos_to_index))

#import ipdb; ipdb.set_trace()
model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT, len(words_to_index), len(pos_to_index))
loss_function = nn.NLLLoss(ignore_index=0)
optimizer = optim.SGD(model.parameters(), lr=0.1)

with torch.no_grad():
    generator = batch_gen(training_indexed, tags_indexed, BATCH_SIZE)
    try:
        sentences_in, sent_lens, targets = next(generator)
        tag_scores = model(sentences_in, sent_lens)

        for idx in range(len(sentences_in[0])):
            if sentences_in[0][idx] < 3:
                continue

            max_idx = 0
            for tag_idx in range(len(tag_scores[0][idx])):
                if tag_scores[0][idx][tag_idx] > tag_scores[0][idx][max_idx]:
                    max_idx = tag_idx

            print('Max score for {} is for tag {}'.format(
                processor.index_to_token[sentences_in[0][idx].item()], 
                processor.index_to_pos[max_idx]))

    except StopIteration:
        print("Nothing")

for epoch in range(NUM_EPOCH):
    generator = batch_gen(training_indexed, tags_indexed, BATCH_SIZE)
    while True:
        try:
            sentences_in, sent_lens, targets = next(generator)

            model.zero_grad()
            tag_scores = model(sentences_in, sent_lens)
            
            total_loss = 0.0
            for idx in range(BATCH_SIZE):
                total_loss += loss_function(tag_scores[idx], targets[idx])
            
            total_loss.backward()
            optimizer.step()

        except StopIteration:
            break

with torch.no_grad():
    print('AFTER TRAINING')

    generator = batch_gen(training_indexed, tags_indexed, BATCH_SIZE)
    try:
        sentences_in, sent_lens, targets = next(generator)
        tag_scores = model(sentences_in, sent_lens)

        for idx in range(len(sentences_in[0])):
            if sentences_in[0][idx] < 3:
                continue

            max_idx = 0
            for tag_idx in range(len(tag_scores[0][idx])):
                if tag_scores[0][idx][tag_idx] > tag_scores[0][idx][max_idx]:
                    max_idx = tag_idx

            print('Max score for {} is for tag {}'.format(
                processor.index_to_token[sentences_in[0][idx].item()], 
                processor.index_to_pos[max_idx]))
                
    except StopIteration:
        print("Nothing")