import torch
from torch import nn
from torch.autograd import Variable
from torch.nn.modules import dropout
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from numpy import prod


class MLP(nn.Module):
    """Module for an MLP with dropout."""
    def __init__(self, input_size, layer_size, depth, activation, dropout):
        super(MLP, self).__init__()
        self.layers = nn.Sequential()
        act_fn = getattr(nn, activation)
        for i in range(depth):
            self.layers.add_module('fc_{}'.format(i),
                                   nn.Linear(input_size, layer_size))
            if activation:
                self.layers.add_module('{}_{}'.format(activation, i),
                                       act_fn())
            if dropout:
                self.layers.add_module('dropout_{}'.format(i),
                                       nn.Dropout(dropout))
            input_size = layer_size

    def forward(self, x):
        return self.layers(x)

class BiAffine(nn.Module):
    """Biaffine attention layer."""
    def __init__(self, input_dim, output_dim):
        super(BiAffine, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.U = nn.Parameter(torch.FloatTensor(output_dim, input_dim, input_dim))
        nn.init.xavier_uniform(self.U)

    def forward(self, Rh, Rd):
        Rh = Rh.unsqueeze(1)
        Rd = Rd.unsqueeze(1)
        S = Rh @ self.U @ Rd.transpose(-1, -2)
        return S.squeeze(1)


class BiAffineParser(nn.Module):
    """Biaffine Dependency Parser."""
    def __init__(self, embeddings, embedding_size,
                 mlp_input, mlp_arc_hidden,
                 mlp_lab_hidden, mlp_dropout,
                 num_labels):
        super(BiAffineParser, self).__init__()

        embeddings = torch.tensor(embeddings)

        self.lstm = nn.LSTM(
            input_size=embeddings.size(1),
            hidden_size=128,
            num_layers=2,
            dropout=0.0,
            bidirectional=True,
        )

        self.hidden_dim = 128
        mlp_input = self.hidden_dim * 2
        # mlp_input = embeddings.size(1)
        #self.embedding = nn.Embedding.from_pretrained(embeddings, padding_idx=0, freeze=False)
        self.embedding = nn.Embedding(embedding_size, 50, padding_idx=0)
        # Arc MLPs
        self.arc_mlp_h = MLP(mlp_input, mlp_arc_hidden, 2, 'ReLU', mlp_dropout)
        self.arc_mlp_d = MLP(mlp_input, mlp_arc_hidden, 2, 'ReLU', mlp_dropout)
        # Label MLPs
        self.lab_mlp_h = MLP(mlp_input, mlp_lab_hidden, 2, 'ReLU', mlp_dropout)

        # BiAffine layers
        self.arc_biaffine = BiAffine(mlp_arc_hidden, 1)
        

        # Loss criterion
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)

    def forward(self, x):

        input = self.embedding(x['sentence'])
        sent_lens = x['lengths']

        embd_input = pack_padded_sequence(input, sent_lens, batch_first=True, enforce_sorted=False)
        h, _ = self.lstm(embd_input.float())
        h, _ = pad_packed_sequence(h, batch_first=True)
        # h = input.float()

        arc_h = self.arc_mlp_h(h)
        arc_d = self.arc_mlp_d(h)

        S_arc = self.arc_biaffine(arc_h, arc_d)
        return S_arc

    def arc_loss(self, S_arc, heads):
        """Compute the loss for the arc predictions."""
        S_arc = S_arc.transpose(-1, -2)                      # [batch, sent_len, sent_len]
        S_arc = S_arc.contiguous().view(-1, S_arc.size(-1))  # [batch*sent_len, sent_len]
        heads = heads.view(-1)                               # [batch*sent_len]
        return self.criterion(S_arc, heads)

    def lab_loss(self, S_lab, heads, labels):
        """Compute the loss for the label predictions on the gold arcs (heads)."""
        heads = heads.unsqueeze(1).unsqueeze(2)              # [batch, 1, 1, sent_len]
        heads = heads.expand(-1, S_lab.size(1), -1, -1)      # [batch, n_labels, 1, sent_len]
        S_lab = torch.gather(S_lab, 2, heads).squeeze(2)     # [batch, n_labels, sent_len]
        S_lab = S_lab.transpose(-1, -2)                      # [batch, sent_len, n_labels]
        S_lab = S_lab.contiguous().view(-1, S_lab.size(-1))  # [batch*sent_len, n_labels]
        labels = labels.view(-1)                             # [batch*sent_len]
        return self.criterion(S_lab, labels)

