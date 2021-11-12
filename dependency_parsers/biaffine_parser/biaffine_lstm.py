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

        self.W = nn.Parameter(torch.Tensor(arc_dim, arc_dim))
        self.reset_parameters()

    def forward(self, head, dep, head_score, dep_score, mask):
        dep_scores = torch.matmul(head, self.W)
        dep_scores = torch.bmm(dep_scores, dep.transpose(1, 2))
        dep_scores += head_score + dep_score.transpose(1, 2)

        mask = torch.bmm(mask.unsqueeze(2).int(), mask.unsqueeze(1).int()).bool()

        # set the scores of arcs incoming to root
        #dep_scores[:, :, 0] = float('-inf')
        # set the scores of self loops n
        #dep_scores.masked_fill_(torch.eye(dep_scores.size(1)).bool(), float('-inf'))

        #dep_scores[~mask] = float('-inf')

        return dep_scores

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W)

class LitLSTM(pl.LightningModule):
    def __init__(self, embeddings, embedding_dim, hidden_dim, num_layers, dropout, arc_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.arc_dim = arc_dim

        self.lstm = LSTM(
            input_size=embedding_dim, 
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=True
        )

        self.word_embedding = nn.Embedding.from_pretrained(torch.tensor(embeddings), padding_idx=0)

        self.hidden2head = Linear(hidden_dim * 2, arc_dim) # this is your g
        self.hidden2dep  = Linear(hidden_dim * 2, arc_dim) # this is your f

        self.head_score = Linear(arc_dim, 1)
        self.dep_score = Linear(arc_dim, 1)

        self.biaffine    = Biaffine(hidden_dim, arc_dim)

        self.alt_loss = nn.CrossEntropyLoss(ignore_index=-100)

        self.log_loss = []
    def loss_function(self, parent_scores, lengths, targets):
        batch, maxlen, _ = parent_scores.shape

        # create masks to work with 
        pads = torch.arange(maxlen) >= lengths.unsqueeze(1)
        mask = pads.unsqueeze(1) | pads.unsqueeze(2)
        rows = pads.unsqueeze(-1).expand((batch, maxlen, maxlen))

        # set the scores of arcs to padding tokens to a large negative number
        parent_scores.masked_fill_(mask, -1e9)
        # set the scores of arcs incoming to root
        parent_scores[:, :, 0] = -1e9
        # set the scores of self loops 
        parent_scores.masked_fill_(torch.eye(maxlen).bool(), -1e9)

        # normalize scores usign softmax
        _S = F.log_softmax(parent_scores, dim=-1)
        S = torch.clone(_S)

        # compute the log partition of the graph using MTT
        Z = self.log_partition(S, lengths)

        S = torch.clone(_S)
        # set the scores of arcs to padding tokens to 0
        S.masked_fill_(mask, 0)
        # set the scores of arcs incoming to root
        S[:, :, 0] = 0
        # set the scores of self loops n
        S.masked_fill_(torch.eye(maxlen).bool(), 0)

        assert maxlen == targets.size(1)

        S = S.reshape((batch * maxlen, maxlen))
        valid = torch.reshape(targets, (batch * maxlen,))

        masker = valid == -100
        offset = torch.arange(start=0, end=maxlen * batch, step=maxlen).unsqueeze(-1).expand(-1, maxlen).reshape((batch * maxlen,))
        indexer = torch.arange(maxlen).repeat(batch, 1).reshape((batch*maxlen,))
        assert valid.shape == indexer.shape
        
        valid[masker] = 0
        valid = valid + offset

        # get the sum of edges of each target tree in the batch
        sums = S[valid, indexer].reshape(batch, maxlen)

        # compute the negative log likelihood of each tree

        P = - (sums.sum(dim=-1) - Z)
        try:
            assert P.mean() > 0
        except AssertionError:
            import ipdb
            ipdb.post_mortem()
        return P.mean()

    def forward(self, x):
        
        sent_lens = x['lengths']

        # Ran: You can make your mask here using sent_lens. The following should do the trick:
        embedding = self.word_embedding(x['sentence'])
        max_len = embedding.shape[1]
        mask = torch.arange(max_len).expand(len(sent_lens), max_len) < sent_lens.unsqueeze(1)
        
        embd_input = pack_padded_sequence(embedding, sent_lens, batch_first=True, enforce_sorted=False)
        
        lstm_out, _ = self.lstm(embd_input.float())
        lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)

        head = F.relu(self.hidden2head(lstm_out))
        dep  = F.relu(self.hidden2dep(lstm_out))

        dep_scores = self.biaffine(head, dep, self.head_score(head), self.dep_score(dep), mask)
        return dep_scores

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=2e-3)
        return optimizer
    
    def training_step(self, train_batch, batch_idx):
        targets = train_batch['parents']
        targets[:, 0] = -100                # we are not interested in the parent of the ROOT TOKEN

        parent_scores = self(train_batch)

        batch, maxlen, _ = parent_scores.shape
        lengths = train_batch['lengths']

        pads = torch.arange(maxlen) >= lengths.unsqueeze(1)
        rows = pads.unsqueeze(-1).expand((batch, maxlen, maxlen))

        #total_loss = self.alt_loss(
        #    parent_scores.reshape((batch * maxlen), maxlen),
        #    targets.reshape((batch * maxlen,))
        #)
        total_loss = self.loss_function(
            parent_scores,
            train_batch['lengths'],
            torch.clone(targets)
        )

        parents = torch.argmax(parent_scores, dim=2)

        num_correct = 0
        total = 0

        num_correct += torch.count_nonzero((parents == targets) * (targets != -100))
        total += torch.count_nonzero((targets == targets) * (targets != -100))

        return {'loss': total_loss, 'correct': num_correct, 'total': total}

    def training_epoch_end(self, outputs):
        correct = 0
        total = 0
        loss = 0.0
        for output in outputs:
            correct += output['correct']
            total += output['total']
            loss += output['loss'] / len(outputs)
        
        self.log_loss.append(loss)
        print('\nAccuracy after epoch end: {:3.3f}'.format(correct/total))

    def validation_step(self, val_batch, batch_idx):
        targets = val_batch['parents']
        targets[:, 0] = -100

        batch, maxlen = targets.shape
        parent_scores = self(val_batch)
        score_clone = torch.clone(parent_scores)

        total_loss = self.loss_function(
            parent_scores,
            val_batch['lengths'],
            torch.clone(targets),
        )

        alt_loss = self.alt_loss(
            score_clone.reshape((batch * maxlen), maxlen),
            targets.reshape((batch * maxlen,))
        )

        parents = torch.argmax(parent_scores, dim=2)

        num_correct = 0
        total = 0

        num_correct += torch.count_nonzero((parents == targets) * (targets != -100))
        total += torch.count_nonzero((targets == targets)* (targets != -100))

        #self.log("train_loss", total_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        return {'loss': total_loss, 'correct': num_correct, 'total': total, 'alt_loss':alt_loss}

    def validation_epoch_end(self, preds):
        correct = 0
        total = 0
        loss = 0
        alt_loss = 0.0
        for pred in preds:
            correct += pred['correct']
            total += pred['total']
            loss += pred['loss']
            alt_loss += pred['alt_loss'] / len(preds)

        print('\nAccuracy on validation set: {:3.3f} | Loss on validation set: {:3.3f} | Cross Entropy: {:3.3f}'.format(correct/total, loss/len(preds), alt_loss))
        return {'accuracy': correct / total, 'loss': loss/len(preds)}

    def test_step(self, batch, batch_idx):
        return dict(self.validation_step(batch, batch_idx))
    
    def test_epoch_end(self, preds):
        return self.validation_epoch_end(preds)


    def log_partition(self, scores, length):
        batch, slen, slen_ = scores.shape
        assert slen == slen_

        pads = torch.arange(slen) >= length.unsqueeze(1)
        mask = pads.unsqueeze(1) | pads.unsqueeze(2)

        # set the scores of arcs to padding tokens to a large negative number
        scores.masked_fill_(mask, -1e9)
        # set the scores of arcs incoming to root
        scores[:, :, 0] = -1e9
        # set the scores of self loops 
        scores.masked_fill_(torch.eye(slen).bool(), -1e9)

        max_score, _ = scores.reshape(batch, -1).max(dim=1)
        weights = (scores - max_score.reshape(batch, 1, 1)).exp() + 1e-8

        weights[:, 0].masked_fill_(pads, 1.)
        w = weights.masked_fill(torch.eye(slen).bool(), 0)

        # Create the Laplacian matrix
        laplacian = -weights
        laplacian.masked_fill_(torch.eye(slen).bool(), 0)
        laplacian += torch.diag_embed(w.sum(dim=1))

        # Compute log partition with MTT
        # 
        # The MTT states that the log partition is equal to the determinant of the matrix 
        # obtained by removing the first row and column from the Laplacian matrix for the weights

        log_part = laplacian[:, 1:, 1:].logdet()
        log_part = log_part + (length.float() - 1) * max_score

        return log_part
