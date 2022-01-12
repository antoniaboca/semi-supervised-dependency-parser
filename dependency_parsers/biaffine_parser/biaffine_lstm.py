from torch._C import set_flush_denormal
from torch.nn.modules import linear
import torchmetrics

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.functional as F

from torch.nn.utils.rnn import pad_sequence, pad_packed_sequence, pack_padded_sequence
from torch.nn import Linear, LSTM

from torch_struct import NonProjectiveDependencyCRF

from stanza.models.common.chuliu_edmonds import chuliu_edmonds_one_root

import pytorch_lightning as pl

class Biaffine(nn.Module):
    def __init__(self, arc_dim, output_dim):
        super().__init__()

        self.W = nn.Parameter(torch.Tensor(output_dim, arc_dim, arc_dim))
        self.reset_parameters()

    def forward(self, head, dep):
        head = head.unsqueeze(1)
        dep = dep.unsqueeze(1)

        scores = head @ self.W @ dep.transpose(-1,-2)
        return scores.squeeze(1)

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W)

class LitLSTM(pl.LightningModule):
    def __init__(self, embeddings, embedding_dim, hidden_dim, num_layers, 
                lstm_dropout, linear_dropout, arc_dim, lab_dim, num_labels, lr, loss_arg, cle_arg):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.arc_dim = arc_dim
        self.lab_dim = lab_dim
        self.lr = lr
        self.cle = cle_arg

        self.lstm = LSTM(
            input_size=embedding_dim, 
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=lstm_dropout,
            bidirectional=True
        )

        self.word_embedding = nn.Embedding.from_pretrained(torch.tensor(embeddings), 
                                                            padding_idx=0)

        # dropout layer
        self.dropout = nn.Dropout(linear_dropout)

        # arc linear layer
        self.arc_linear_h = Linear(hidden_dim * 2, arc_dim)  # this is your g
        self.arc_linear_d = Linear(hidden_dim * 2, arc_dim) # this is your f

        #label linear layer 
        self.lab_linear_h = Linear(hidden_dim * 2, lab_dim)
        self.lab_linear_d = Linear(hidden_dim * 2, lab_dim)

        #arc scores
        self.arc_score_h = Linear(arc_dim, 1)
        self.arc_score_d = Linear(arc_dim, 1)

        #lab scores
        self.lab_score_h = Linear(lab_dim, num_labels)
        self.lab_score_d = Linear(lab_dim, num_labels)

        # biaffine layers
        self.arc_biaffine = Biaffine(arc_dim, 1)
        self.lab_biaffine = Biaffine(lab_dim, num_labels)

        if loss_arg == 'cross':
            self.loss = nn.CrossEntropyLoss(ignore_index=-1)
        elif loss_arg == 'mtt':
            self.loss = self.loss_function

        self.log_loss = []
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
    
    def forward(self, x):
        lengths = x['lengths']

        embedding = self.word_embedding(x['sentence'])
        maxlen = embedding.shape[1]

        # mask = torch.arange(maxlen).expand(len(lengths), maxlen) < lengths.unsqueeze(1)
        
        embd_input = pack_padded_sequence(embedding, lengths, batch_first=True, enforce_sorted=False)
        
        lstm_out, _ = self.lstm(embd_input.float())
        lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)
        
        # arcs
        h_arc = self.dropout(F.relu(self.arc_linear_h(lstm_out)))
        d_arc = self.dropout(F.relu(self.arc_linear_d(lstm_out)))

        # labels
        h_lab = self.dropout(F.relu(self.lab_linear_h(lstm_out)))
        d_lab = self.dropout(F.relu(self.lab_linear_d(lstm_out)))

        # arc scores
        h_score_arc = self.arc_score_h(h_arc)
        d_score_arc = self.arc_score_d(d_arc)

        # label scores
        #h_score_lab = self.lab_score_h(h_lab)
        #d_score_lab = self.lab_score_d(d_lab)

        arc_scores = self.arc_biaffine(h_arc, d_arc) + h_score_arc + d_score_arc.transpose(1, 2)
        lab_scores = self.lab_biaffine(h_lab, d_lab)

        return arc_scores, lab_scores
        
    def training_step(self, train_batch, batch_idx):
        parents = train_batch['parents']
        labels = train_batch['labels']

        arc_scores, lab_scores = self(train_batch)
        batch, maxlen, _ = arc_scores.shape
        lengths = train_batch['lengths']

        arc_loss = self.arc_loss(arc_scores, parents)
        lab_loss = self.lab_loss(lab_scores, parents, labels)

        total_loss = arc_loss + lab_loss
        targets = torch.argmax(arc_scores, dim=2)

        num_correct = 0
        total = 0

        num_correct += torch.count_nonzero((targets == parents) * (parents != -1))
        total += torch.count_nonzero((parents == parents) * (parents != -1))
        
        self.log('training_loss', total_loss.detach(), on_step=True, on_epoch=True, logger=True)
        self.log('training_arc_loss', arc_loss.detach(), on_step=True, on_epoch=True, logger=True)
        self.log('training_lab_loss', lab_loss.detach(), on_step=True, on_epoch=True, logger=True)

        return {'loss': total_loss, 
                'correct': num_correct, 
                'total': total, 
                'arc_loss': arc_loss.detach(), 
                'lab_loss': lab_loss.detach()
        }
        
    def training_epoch_end(self, outputs):
        correct = 0
        total = 0
        loss = 0.0
        arc_loss = 0.0
        lab_loss = 0.0
        for output in outputs:
            correct += output['correct']
            total += output['total']
            loss += output['loss'] / len(outputs)
            arc_loss += output['arc_loss'] / len(outputs)
            lab_loss += output['lab_loss'] / len(outputs)
        
        self.log_loss.append(loss)

        self.log('training_accuracy', correct/total, on_epoch=True, on_step=False, logger=True)
        print('\nAccuracy after epoch end: {:3.3f} | Arc loss: {:3.3f} | Lab loss: {:3.3f}'.format(correct/total, arc_loss.detach(), lab_loss.detach()))

    def validation_step(self, val_batch, batch_idx):
        targets = val_batch['parents']
        labels = val_batch['labels']
        
        batch, maxlen = targets.shape
        arc_scores, lab_scores = self(val_batch)

        arc_loss = self.arc_loss(arc_scores,targets)
        lab_loss = self.lab_loss(lab_scores, targets, labels)

        total_loss = arc_loss + lab_loss
        num_correct = 0
        total = 0
        
        parents = torch.argmax(arc_scores, dim=2)
        num_correct += torch.count_nonzero((parents == targets) * (targets != -1))
        total += torch.count_nonzero((targets == targets)* (targets != -1))

        self.log('validation_loss', total_loss.detach(), on_step=False, on_epoch=True, logger=True)
        self.log('validation_arc_loss', arc_loss.detach(), on_step=False, on_epoch=True, logger=True)
        self.log('validation_lab_loss', lab_loss.detach(), on_step=False, on_epoch=True, logger=True)

        return {'loss': total_loss, 'correct': num_correct, 'total': total, 'arc_loss':arc_loss.detach(), 'lab_loss': lab_loss.detach()}

    def validation_epoch_end(self, preds):
        correct = 0
        total = 0
        loss = 0
        arc_loss = 0.0
        lab_loss = 0.0
        for pred in preds:
            correct += pred['correct']
            total += pred['total']
            loss += pred['loss']
            arc_loss += pred['arc_loss'] / len(preds)
            lab_loss += pred['lab_loss'] / len(preds)

        self.log("validation_accuracy",correct/total, on_epoch=True, logger=True)

        print('\nAccuracy on validation set: {:3.3f} | Loss : {:3.3f} | Arc loss: {:3.3f} | Lab loss: {:3.3f}'.format(correct/total, loss/len(preds), arc_loss.detach(), lab_loss.detach()))
        return {'accuracy': correct / total, 'loss': loss/len(preds)}

    def test_step(self, test_batch, batch_idx):
        targets = test_batch['parents']
        labels = test_batch['labels']
        lengths = test_batch['lengths']

        batch, maxlen = targets.shape
        arc_scores, lab_scores = self(test_batch)
        
        if self.cle:
            trees, arc_loss = self.edmonds_arc_loss(arc_scores, lengths, targets)
        else:
            trees = torch.argmax(arc_scores, dim=2)
            arc_loss = self.arc_loss(arc_scores, targets)

        lab_loss = self.lab_loss(lab_scores, targets, labels)

        total_loss = arc_loss + lab_loss
        num_correct = 0
        total = 0

        num_correct += torch.count_nonzero((trees == targets) * (targets != -1))
        total += torch.count_nonzero((targets == targets)* (targets != -1))

        self.log('test_loss', total_loss.detach(), on_step=False, on_epoch=True, logger=True)
        self.log('test_arc_loss', arc_loss.detach(), on_step=False, on_epoch=True, logger=True)
        self.log('test_lab_loss', lab_loss.detach(), on_step=False, on_epoch=True, logger=True)

        return {'loss': total_loss, 'correct': num_correct, 'total': total, 'arc_loss':arc_loss.detach(), 'lab_loss': lab_loss.detach()}

    
    def test_epoch_end(self, preds):
        return self.validation_epoch_end(preds)

    def lab_loss(self, S_lab, parents, labels):
        """Compute the loss for the label predictions on the gold arcs (heads)."""
        heads = torch.clone(parents)
        heads[heads == -1] = 0
        heads = heads.unsqueeze(1).unsqueeze(2)              # [batch, 1, 1, sent_len]
        heads = heads.expand(-1, S_lab.size(1), -1, -1)      # [batch, n_labels, 1, sent_len]
        S_lab = torch.gather(S_lab, 2, heads).squeeze(2)     # [batch, n_labels, sent_len]
        S_lab = S_lab.transpose(-1, -2)                      # [batch, sent_len, n_labels]
        S_lab = S_lab.contiguous().view(-1, S_lab.size(-1))  # [batch*sent_len, n_labels]
        labels = labels.view(-1)                             # [batch*sent_len]
        return self.loss(S_lab, labels)

    def arc_loss(self, S_arc, heads):
        """Compute the loss for the arc predictions."""
        #S_arc = S_arc.transpose(-1, -2)                      # [batch, sent_len, sent_len]
        S_arc = S_arc.contiguous().view(-1, S_arc.size(-1))  # [batch*sent_len, sent_len]
        heads = heads.view(-1)                               # [batch*sent_len]
        return self.loss(S_arc, heads)
    
    def edmonds_arc_loss(self, S_arc, lengths, heads):
        S = torch.clone(S_arc.detach())
        batch_size = S.size(0)
        trees = []

        for batch in range(batch_size):
            length = lengths[batch]
            graph = S[batch][:length, :length]
            tree = chuliu_edmonds_one_root(graph.numpy())
            trees.append(torch.tensor(tree))

        batched = pad_sequence(trees, batch_first=True, padding_value=-1)
        batched[:, 0] = -1

        return batched, self.arc_loss(S_arc, heads)

    def new_loss(self, arc_scores, lengths, targets):
        batch, maxlen, _ = arc_scores.shape
        S = NonProjectiveDependencyCRF(arc_scores, lengths)

        S = S.reshape((batch * maxlen, maxlen))
        valid = torch.reshape(targets, (batch * maxlen,))

        offset = torch.arange(start=0, end=maxlen * batch, step=maxlen).unsqueeze(-1).expand(-1, maxlen).reshape((batch * maxlen,))
        indexer = torch.arange(maxlen).repeat(batch, 1).reshape((batch*maxlen,))
        assert valid.shape == indexer.shape
        
        valid = valid + offset

        # get the sum of edges of each target tree in the batch
        sums = S[valid, indexer].reshape(batch, maxlen)

        # compute the negative log likelihood of each tree

        P = sums.sum(dim=-1)

        return P

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

        masker = valid == -1
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
