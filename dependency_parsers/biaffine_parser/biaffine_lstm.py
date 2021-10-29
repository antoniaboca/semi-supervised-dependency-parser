import torchmetrics

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.nn.utils.rnn import pad_sequence, pad_packed_sequence, pack_padded_sequence
from torch.nn import Linear, LSTM

import pytorch_lightning as pl

class G(pl.LightningModule):
    def __init__(self, input_dim, arc_dim):
        super().__init__()

        self.input_dim = input_dim
        self.arc_dim = arc_dim

        self.g = nn.Parameter(torch.Tensor(arc_dim, input_dim))

        self.reset_parameters()

    def forward(self, x):
        return torch.einsum('bij,bjk->bik', x, self.g)

    def reset_parameters(self):
        nn.init.zeros_(self.g)

class Biaffine(pl.LightningModule):
    def __init__(self, input_dim, output_dim, arc_dim, scale=0):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.scale = scale
        
        self.W = nn.Parameter(torch.Tensor(output_dim, arc_dim, arc_dim))

        self.reset_parameters()

    def forward(self, x, y):
        return torch.einsum('bij,jk,bkt->bij', x, self.W, y)

    def reset_parameters(self):
        nn.init.zeros_(self.W)

class LitLSTM(pl.LightningModule):
    def __init__(self, embedding_dim, hidden_dim, num_layers, dropout, arc_dim):
        super(LitLSTM, self).__init__()
        self.hidden_dim = hidden_dim

        self.lstm = LSTM(
            input_size=embedding_dim, 
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=True
        )

        self.hidden2head = Linear(hidden_dim * 2, arc_dim)
        self.hidden2dep  = Linear(hidden_dim * 2, arc_dim)
        self.biaffine    = Biaffine(arc_dim, 1, arc_dim)
        self.g           = G(1, arc_dim)

        self.loss_function = nn.CrossEntropyLoss(ignore_index=-100)
        
        self.loss_fn = []

    def forward(self, x):
        
        sent_lens = x['lengths']

        #sent_input = pack_padded_sequence(x['sentence'], sent_lens, batch_first=True, enforce_sorted=False)
        embd_input = pack_padded_sequence(x['embedding'], sent_lens, batch_first=True, enforce_sorted=False)
        #tags_input = pack_padded_sequence(x['tags'], sent_lens, batch_first=True, enforce_sorted=False)
        
        lstm_out, _ = self.lstm(embd_input.float())
        lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)
        
        heads = self.hidden2head(lstm_out)
        deps  = self.hidden2dep(lstm_out)
        
        eeeh = self.g(heads) + self.g(deps) + self.biaffine(heads, deps)
        scores = []
        for idx in range(len(lstm_out)):
            sequence = lstm_out[idx]
            dep_score = torch.zeros(len(sequence), len(sequence))
            for i in range(len(sequence)):
                for j in range(len(sequence)):
                    sij = self.g(heads[ idx ][ i ]) + self.g(deps[ idx ][ j ]) + self.biaffine(heads[ idx ][ i ], deps[ idx ][ j ])
                    dep_score[ i ][ j ] = sij
            
            max_score = torch.argmax(dep_score, dim=1)
            scores.append(dep_score)

        return pad_sequence(scores, batch_first=True)
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=0.05)
        return optimizer
    
    def training_step(self, train_batch, batch_idx):
        targets = train_batch['parents']

        parent_scores = self(train_batch)

        print(parent_scores.shape)
        batch_size, sent_len, score_len = parent_scores.shape
        total_loss = self.loss_function(
            parent_scores.reshape(batch_size * sent_len, score_len),
            targets.reshape(batch_size * sent_len)
        )

        parents = torch.argmax(parent_scores, dim=2)

        num_correct = 0
        total = 0

        num_correct += torch.count_nonzero((parents == targets) * (targets != -100))
        total += torch.count_nonzero(targets)

        #self.log("train_loss", total_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return {'loss': total_loss, 'correct': num_correct, 'total': total}

    def training_epoch_end(self, outputs):
        correct = 0
        total = 0
        for output in outputs:
            correct += output['correct']
            total += output['total']
            self.loss_fn.append(output['loss'])
        
        print('Accuracy after epoch end: {}'.format(correct/total))

    def validation_step(self, batch, batch_idx):
        targets = batch['parents']

        parent_scores = self(batch)

        batch_size, sent_len, num_tags = parent_scores.shape
        total_loss = self.loss_function(
            parent_scores.reshape(batch_size * sent_len, num_tags),
            targets.reshape(batch_size * sent_len)
        )

        parents = torch.argmax(parent_scores, dim=2)

        num_correct = 0
        total = 0

        num_correct += torch.count_nonzero((parents == targets) * (targets != -100))
        total += torch.count_nonzero(targets)

        #self.log("train_loss", total_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return {'loss': total_loss, 'correct': num_correct, 'total': total}

    def validation_epoch_end(self, preds):
        correct = 0
        total = 0
        loss = 0

        for pred in preds:
            correct += pred['correct']
            total += pred['total']
            loss += pred['loss']

        self.log('accuracy', correct / total)
        self.log('loss', loss/len(preds))

        print('Accuracy on validation set: {} | Loss on validation set: {}'.format(correct/total, loss/len(preds)))
        return {'accuracy': correct / total, 'loss': loss/len(preds)}

    def test_step(self, batch, batch_idx):
        return dict(self.validation_step(batch, batch_idx))
    
    def test_epoch_end(self, preds):
        return self.validation_epoch_end(preds)
