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

from dependency_parsers.biaffine_parser.biaffine_lstm import LitLSTM

class LitSemiTransferLSTM(pl.LightningModule):
    def __init__(self, args, prior):
        super().__init__()
        self.loss = nn.CrossEntropyLoss(ignore_index=-1)
        self.lr = args.lr
        self.cle = args.cle
        self.prior = prior
        self.feature_extractor = LitLSTM.load_from_checkpoint('my_logs/default/version_63/checkpoints/epoch=15-step=511.ckpt', 
            hparams_file='my_logs/default/version_63/hparams.yaml')

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
    
    def forward(self, x):
        return self.feature_extractor(x)
        
    def training_step(self, batch, batch_idx):
        unlabelled = batch['unlabelled']
        labelled = batch['labelled']

        features = unlabelled['features']
        #parents = unlabelled['parents']
        #labels = unlabelled['labels']

        # COMPUTE LOSS FOR UNLABELLED DATA
        arc_scores, lab_scores = self(unlabelled)
        batch_unlabelled, maxlen_unlabelled, _ = arc_scores.shape
        lengths = unlabelled['lengths']
        
        potentials = self.score_to_diagonal(arc_scores)
        log_potentials = self.apply_log_softmax(potentials, lengths)
        dist = NonProjectiveDependencyCRF(log_potentials, lengths - torch.ones(len(lengths)))
    
        assert not torch.isnan(dist.partition).any()
        assert not torch.isinf(dist.partition).any()
        assert not torch.isinf(dist.marginals).any()
        assert not torch.isnan(dist.marginals).any()
        
        #Z = dist.partition
        #total_loss = self.partition_loss(torch.clone(potentials), torch.clone(parents), lengths, Z)
        diag_features = self.feature_to_diagonal(features.float())
        unlabelled_loss = self.GE_loss(dist.marginals, diag_features).mean()

        # arc_loss = self.arc_loss(arc_scores, parents)
        # lab_loss = self.lab_loss(lab_scores, parents, labels)
        # total_loss = arc_loss + lab_loss

        #self.log('training_arc_loss', arc_loss.detach(), on_step=True, on_epoch=True, logger=True)
        #self.log('training_lab_loss', lab_loss.detach(), on_step=True, on_epoch=True, logger=True)

        
        return {'loss': unlabelled_loss, 
                # 'arc_loss': arc_loss.detach(), 
                #'lab_loss': lab_loss.detach()
        }
        
    def training_epoch_end(self, outputs):
        correct = 0
        total = 0
        loss = 0.0
        labelled = 0.0
        unlabelled = 0.0
        for output in outputs:
            loss += output['loss'] / len(outputs)
            # arc_loss += output['arc_loss'] / len(outputs)
            # lab_loss += output['lab_loss'] / len(outputs)

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

    def GE_loss(self, scores, features):
        X = torch.einsum('bij,bijf->bf', scores, features)
        Y = X - self.prior
        return 0.5 * (Y.unsqueeze(-2) @ Y.unsqueeze(-1)).squeeze(-1).squeeze(-1)

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

    def apply_log_softmax(self, scores, lengths):
        lengths = lengths - torch.ones(len(lengths))

        batch, maxlen, _ = scores.shape
        pads = torch.arange(maxlen) >= lengths.unsqueeze(1)
        mask = pads.unsqueeze(1) | pads.unsqueeze(2)

        # set the scores of arcs to padding tokens to a large negative number
        scores.masked_fill_(mask, -1e9)
        return F.log_softmax(scores, dim=-1)

    def feature_to_diagonal(self, features):
        """
            Turns the feature vector (N+1, N+1, 20) where ROOT is index 0
            into a feature vector (N, N, 20) where diagonals represent
            outgoing edges from the root
        """
        batch, maxlen, _, dim = features.shape

        outgoing = features[:, 0, 1:, :]
        shrink = features[:, 1:, 1:, :]
        shrink[:, range(maxlen-1), range(maxlen-1), :] = outgoing
        return shrink

    def score_to_diagonal(self, scores):
        """
            Turning a matrix of scores (N+1, N+1) where ROOT is index 0
            into a matrix of scores (N, N) where diagonals represent
            outgoing edges from the root

            e.g. a[ i ][ i ] = score(ROOT -> i)
        """
        batch, maxlen, _ = scores.shape

        outgoing = scores[:, 0, 1:]
        shrink = scores[:, 1:, 1:]
        shrink[:, range(maxlen-1), range(maxlen-1)] = outgoing
        return shrink

    def partition_loss(self, scores, targets, lengths, Z):
        # using scores with ROOT on diagonal
        lengths = lengths - torch.ones(len(lengths))
        targets = targets[:, 1:]
        targets = targets - torch.ones(targets.shape)

        # set the scores of arcs to padding tokens to 0
        batch, maxlen, _ = scores.shape
        pads = torch.arange(maxlen) >= lengths.unsqueeze(1)
        mask = pads.unsqueeze(1) | pads.unsqueeze(2)

        scores.masked_fill_(mask, 0)

        # modify targets such that ROOT is on the diagonal
        ends = torch.arange(maxlen) >= lengths.unsqueeze(1)
        mask = torch.logical_and(targets == -1, ~ends)
        vals = torch.arange(maxlen).expand(batch, -1)
        targets[mask] = vals[mask].float()  
        targets[targets == -2] = -1
        
        batch_idxs = []
        row_idxs = []
        col_idxs = []
        
        sums = torch.tensor(0.0)

        for idx in range(batch):
            row_idxs = targets[idx][:lengths[idx].int()].long()
            col_idxs = torch.tensor(range(lengths[idx].int())).long()
            t = torch.tensor(idx)
            batch_idxs = t.repeat(lengths[idx].int()).long()
            sum = scores[batch_idxs, row_idxs, col_idxs].reshape(lengths[idx].int()).sum(dim=-1)
            sums += (-(sum - Z[idx])) / batch

        return sums
        
