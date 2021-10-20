import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils.rnn import pad_packed_sequence
from torch.nn import Linear, LSTM

class LSTMTagger(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_layers, dropout, tagset_size):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim

        self.lstm = LSTM(
            input_size=embedding_dim, 
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=True
        )

        self.hidden2tag = Linear(hidden_dim * 2, tagset_size)
    
    def forward(self, x):
        embedding = x['embedding']
        
        lstm_out, _ = self.lstm(embedding.float())
        lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)
        
        tag_scores = self.hidden2tag(lstm_out)
        return tag_scores
