import torchmetrics

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.functional as F

from torch.nn.utils.rnn import pad_sequence, pad_packed_sequence, pack_padded_sequence
from torch.nn import Linear, LSTM

import pytorch_lightning as pl

class Biaffine(nn.Module):
    def __init__(self, hidden_dim, arc_dim):
        super().__init__()

        self.W = nn.Parameter(torch.rand(arc_dim, arc_dim))
        self.hidden2head = Linear(hidden_dim * 2, arc_dim) # this is your g
        self.hidden2dep  = Linear(hidden_dim * 2, arc_dim) # this is your f

        self.reset_parameters()

    def forward(self, lstm_out, mask):
        head = self.hidden2head(lstm_out)
        dep  = self.hidden2dep(lstm_out)
        
        mask = mask.unsqueeze(-1).expand(-1, -1, head.size(2))
        head[~mask] = 0.0
        dep[~mask]  = 0.0

        return torch.einsum('bij,jk,bkt->bit', head, self.W, torch.transpose(dep, 1, 2))

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W)

class LitLSTM(pl.LightningModule):
    def __init__(self, embedding_dim, hidden_dim, num_layers, dropout, arc_dim):
        super(LitLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.arc_dim = arc_dim

        self.lstm = LSTM(
            input_size=embedding_dim, 
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=True
        )
        
        self.biaffine    = Biaffine(hidden_dim, arc_dim)
        self.loss_function = nn.CrossEntropyLoss(ignore_index=-100)
        
    def forward(self, x):
        
        sent_lens = x['lengths']

        # Ran: You can make your mask here using sent_lens. The following should do the trick:
        embeddings = x['embedding']
        max_len = embeddings.shape[1]
        mask = torch.arange(max_len).expand(len(sent_lens), max_len) < sent_lens.unsqueeze(1)
        
        embd_input = pack_padded_sequence(x['embedding'], sent_lens, batch_first=True, enforce_sorted=False)
        
        lstm_out, _ = self.lstm(embd_input.float())
        lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)

        dep_scores = self.biaffine(lstm_out, mask)
        return dep_scores

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=0.05)
        return optimizer
    
    def training_step(self, train_batch, batch_idx):
        targets = train_batch['parents']
        targets[:, 0] = -100                # we are not interested in the parent of the ROOT TOKEN

        parent_scores = self(train_batch)

        batch_size, sent_len, score_len = parent_scores.shape
        total_loss = self.loss_function(
            parent_scores.reshape(batch_size * sent_len, score_len),
            targets.reshape(batch_size * sent_len)
        )

        parents = torch.argmax(parent_scores, dim=2)

        num_correct = 0
        total = 0

        num_correct += torch.count_nonzero((parents == targets) * (targets != -100))
        total += torch.count_nonzero(targets * (targets != -100))

        return {'loss': total_loss, 'correct': num_correct, 'total': total}

    def training_epoch_end(self, outputs):
        correct = 0
        total = 0
        for output in outputs:
            correct += output['correct']
            total += output['total']
        
        print('Accuracy after epoch end: {}'.format(correct/total))

    def validation_step(self, batch, batch_idx):
        targets = batch['parents']
        targets[:, 0] = -100
        
        parent_scores = self(batch)

        batch_size, sent_len, _ = parent_scores.shape
        total_loss = self.loss_function(
            parent_scores.reshape(batch_size * sent_len, sent_len),
            targets.reshape(batch_size * sent_len)
        )

        parents = torch.argmax(parent_scores, dim=2)

        num_correct = 0
        total = 0

        num_correct += torch.count_nonzero((parents == targets) * (targets != -100))
        total += torch.count_nonzero(targets * (targets != -100))

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
