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
FILE_SIZE = 100
NUM_EPOCH = 10

def prepare_sequence(seq, to_index):
    idxs = [to_index[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)


processor = DataProcessor(UD_ENGLISH_GUM, FILE_SIZE)

words_to_index = processor.word_to_index
pos_to_index = processor.pos_to_index
training_data = processor.sentences

class LSTMTagger(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_layers, dropout, vocab_size, tagset_size):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = Embedding(vocab_size, embedding_dim)

        self.lstm = LSTM(
            input_size=embedding_dim, 
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=True
        )

        self.hidden2tag = Linear(hidden_dim * 2, tagset_size)
    
    def forward(self, sentence1, sentence2):
        embed1 = self.word_embeddings(sentence1)
        embed2 = self.word_embeddings(sentence2)
        padded = pad_sequence([embed1, embed2], batch_first=True)
        sent_lens = torch.tensor([len(sentence1), len(sentence2)])

        lstm_input = pack_padded_sequence(padded, sent_lens, batch_first=True, enforce_sorted=False)
        lstm_out, _ = self.lstm(lstm_input)
        lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)
        
        tag_space = self.hidden2tag(lstm_out)
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores

#import ipdb; ipdb.set_trace()
model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT, len(words_to_index), len(pos_to_index))
loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

with torch.no_grad():
    sentence1, sentence2 = training_data[0][0], training_data[1][0]
    input1, input2 = prepare_sequence(sentence1, words_to_index), prepare_sequence(sentence2, words_to_index)
    tag_scores = model(input1, input2)
    #print(tag_scores)

    for idx in range(len(sentence1)):
        max_idx = 0
        for tag_idx in range(len(tag_scores[0][idx])):
            if tag_scores[0][idx][tag_idx] > tag_scores[0][idx][max_idx]:
                max_idx = tag_idx

        print('Max score for {} is for tag {}'.format(
            sentence1[idx], 
            processor.index_to_pos[max_idx]))

for epoch in range(NUM_EPOCH):
    for idx in range(len(training_data)):
        if idx + 1 == len(training_data):
            break

        sentence1, tags1 = training_data[idx]
        sentence2, tags2 = training_data[idx + 1]

        model.zero_grad()
        
        sentence_in_1 = prepare_sequence(sentence1, words_to_index)
        targets1 = prepare_sequence(tags1, pos_to_index)

        sentence_in_2 = prepare_sequence(sentence2, words_to_index)
        targets2 = prepare_sequence(tags2, pos_to_index)

        tag_scores = model(sentence_in_1, sentence_in_2)
        
        #import ipdb; ipdb.set_trace()

        targets = pad_sequence([targets1, targets2], batch_first=True)
        total_loss = loss_function(tag_scores[0], targets[0]) + loss_function(tag_scores[1], targets[1])
        total_loss.backward()
        optimizer.step()

with torch.no_grad():
    sentence1, sentence2 = training_data[0][0], training_data[1][0]
    input1, input2 = prepare_sequence(sentence1, words_to_index), prepare_sequence(sentence2, words_to_index)
    tag_scores = model(input1, input2)
    #print(tag_scores)
    print("AFTER TRAINING")
    
    for idx in range(len(sentence1)):
        max_idx = 0
        for tag_idx in range(len(tag_scores[0][idx])):
            if tag_scores[0][idx][tag_idx] > tag_scores[0][idx][max_idx]:
                max_idx = tag_idx

        print('Max score for {} is for tag {}'.format(
            sentence1[idx], 
            processor.index_to_pos[max_idx]))