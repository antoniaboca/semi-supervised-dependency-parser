from cmath import inf
from torch._C import set_flush_denormal
from torch.nn.modules import linear
from dependency_parsers.data.processor import PAD_VALUE, unlabelled_padder
import torchmetrics

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from dependency_parsers.biaffine_parser.biaffine_lstm import LitLSTM
from dependency_parsers.nn.layers import Biaffine, MLP

from torch.nn.utils.rnn import pad_sequence, pad_packed_sequence, pack_padded_sequence
from torch.nn import Linear, LSTM

from torch_struct import NonProjectiveDependencyCRF

from stanza.models.common.chuliu_edmonds import chuliu_edmonds_one_root

import pytorch_lightning as pl

class LitEntropyLSTM(pl.LightningModule):
    def __init__(self, embeddings, f_star, args, loss, num_labels):
        super().__init__()

        if loss == 'cross':
            self.loss = nn.CrossEntropyLoss(ignore_index=-1)
        elif loss == 'mtt':
            self.loss = self.loss_function

        self.transfer = args.transfer
        self.log_softmax = nn.LogSoftmax(dim=-1)
        self.ge_only = args.ge_only

        if args.transfer is True:
            self.lr = args.lr
            self.cle = args.cle
            self.prior = f_star
            self.model = LitLSTM.load_from_checkpoint('my_logs/default/version_73/checkpoints/epoch=15-step=1007.ckpt', 
                            hparams_file='my_logs/default/version_73/hparams.yaml')
            return    
    
        self.hidden_dim = args.hidden_dim
        self.arc_dim = args.arc_dim
        self.lab_dim = args.lab_dim
        self.lr = args.lr
        self.cle = args.cle
        self.prior = f_star
        

        self.lstm = LSTM(
            input_size=args.embedding_dim, 
            hidden_size=args.hidden_dim,
            num_layers=args.num_layers,
            dropout=args.lstm_dropout,
            bidirectional=True
        )

        self.word_embedding = nn.Embedding.from_pretrained(torch.tensor(embeddings), 
                                                            padding_idx=0)

        # arc and label MLP layers
        self.linear_dropout = args.linear_dropout

        self.MLP_arc = MLP(self.hidden_dim * 2, self.linear_dropout, self.arc_dim)
        self.MLP_lab = MLP(self.hidden_dim * 2, self.linear_dropout, self.lab_dim)

        # biaffine layers
        self.arc_biaffine = Biaffine(self.arc_dim, 1)
        self.lab_biaffine = Biaffine(self.lab_dim, num_labels)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
    
    def forward(self, x):
        if self.transfer:
            return self.model(x)

        lengths = x['lengths']
        embedding = self.word_embedding(x['sentence'])
        maxlen = embedding.shape[1]

        embd_input = pack_padded_sequence(embedding, lengths, batch_first=True, enforce_sorted=False)
        lstm_out, _ = self.lstm(embd_input.float())
        lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)
        
        h_arc, d_arc, h_score_arc, d_score_arc = self.MLP_arc(lstm_out)
        h_lab, d_lab, _, _ = self.MLP_lab(lstm_out)

        arc_scores = self.arc_biaffine(h_arc, d_arc) + h_score_arc + d_score_arc.transpose(1, 2)
        lab_scores = self.lab_biaffine(h_lab, d_lab)

        return arc_scores, lab_scores
        
    def training_step(self, batch, batch_idx):
        unlabelled = batch['unlabelled']
        features = unlabelled['features']

        # COMPUTE LOSS FOR UNLABELLED DATA
        arc_scores, lab_scores = self(unlabelled)
        batch_unlabelled, maxlen_unlabelled, _ = arc_scores.shape
        lengths = unlabelled['lengths']
        
        potentials = self.score_to_diagonal(arc_scores)
        log_potentials = self.apply_log_softmax(potentials, lengths)

        try:
            assert torch.all(log_potentials <= 0)
        except AssertionError:
            import ipdb; ipdb.post_mortem()
        
        dist = NonProjectiveDependencyCRF(log_potentials, lengths - torch.ones(len(lengths)))

        assert not torch.isnan(dist.partition).any()
        assert not torch.isinf(dist.partition).any()
        assert not torch.isinf(dist.marginals).any()
        assert not torch.isnan(dist.marginals).any()

        #Z = dist.partition
        #total_loss = self.partition_loss(torch.clone(potentials), torch.clone(parents), lengths, Z)
        #diag_features = self.feature_to_diagonal(features.float())
        unlabelled_loss = self.entropy_loss(dist.marginals, torch.clone(-log_potentials))

        self.log('unlabelled_loss', unlabelled_loss.detach(), on_step=True, on_epoch=True, logger=True)

        if self.ge_only:
            self.log('training_loss', unlabelled_loss.detach(), on_step=True, on_epoch=True, logger=True)
            return {
                'loss': unlabelled_loss,
                'unlabelled_loss': unlabelled_loss,
            }

        # COMPUTE LOSS FOR LABELLED DATA
        labelled = batch['labelled']
        l_arc_scores, l_lab_scores = self(labelled)
        batch_labelled, maxlen_labelled, _ = l_arc_scores.shape
        lengths = labelled['lengths']
        parents = labelled['parents']

        labelled_loss = self.arc_loss(l_arc_scores, parents)
        # lab_loss = self.lab_loss(lab_scores, parents, labels)

        batch_total = batch_labelled + batch_unlabelled
        total_loss = (batch_labelled / batch_total) * labelled_loss + (batch_unlabelled / batch_total) * unlabelled_loss

        targets = torch.argmax(l_arc_scores, dim=2)

        num_correct = 0
        total = 0

        num_correct += torch.count_nonzero((targets == parents) * (parents != -1))
        total += torch.count_nonzero((parents == parents) * (parents != -1))
        
        self.log('training_loss', total_loss.detach(), on_step=True, on_epoch=True, logger=True)
        #self.log('training_arc_loss', arc_loss.detach(), on_step=True, on_epoch=True, logger=True)
        #self.log('training_lab_loss', lab_loss.detach(), on_step=True, on_epoch=True, logger=True)
        
        return {'loss': total_loss, 
                'correct': num_correct, 
                'total': total, 
                'labelled_loss': labelled_loss.detach(),
                'unlabelled_loss': unlabelled_loss.detach(),
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

            if not self.ge_only:
                labelled += output['labelled_loss'] / len(outputs)
                unlabelled += output['unlabelled_loss'] / len(outputs)
                correct += output['correct']
                total += output['total']
            
        if not self.ge_only:
            self.log('training_accuracy', correct/total, on_epoch=True, on_step=False, logger=True)
            print('\nAccuracy on labelled data: {:3.3f} | Unlabelled loss: {:3.3f} | Labelled loss: {:3.3f}'.format(correct/total, unlabelled, labelled))

    def validation_step(self, val_batch, batch_idx):
        targets = val_batch['parents']
        labels = val_batch['labels']
        
        arc_scores, lab_scores = self(val_batch)
        batch_unlabelled, maxlen_unlabelled, _ = arc_scores.shape
        lengths = val_batch['lengths']
        
        potentials = self.score_to_diagonal(arc_scores)
        log_potentials = self.apply_log_softmax(potentials, lengths)

        try:
            assert torch.all(log_potentials <= 0)
        except AssertionError:
            #import ipdb; ipdb.post_mortem()
            pass
        
        dist = NonProjectiveDependencyCRF(log_potentials, lengths - torch.ones(len(lengths)))

        assert not torch.isnan(dist.partition).any()
        assert not torch.isinf(dist.partition).any()
        assert not torch.isinf(dist.marginals).any()
        assert not torch.isnan(dist.marginals).any()

        #Z = dist.partition
        #total_loss = self.partition_loss(torch.clone(potentials), torch.clone(parents), lengths, Z)
        #diag_features = self.feature_to_diagonal(features.float())
        unlabelled_loss = self.entropy_loss(dist.marginals, torch.clone(-log_potentials))

        self.log('unlabelled_loss', unlabelled_loss.detach(), on_step=True, on_epoch=True, logger=True)

        #if self.ge_only:
        #    return {
        #        'loss': unlabelled_loss
        #    }

        # COMPUTE LOSS FOR LABELLED DATA
        l_arc_scores, l_lab_scores = self(val_batch)
        batch_labelled, maxlen_labelled, _ = l_arc_scores.shape
        lengths = val_batch['lengths']
        parents = val_batch['parents']

        labelled_loss = self.arc_loss(l_arc_scores, parents)
        # lab_loss = self.lab_loss(lab_scores, parents, labels)

        batch_total = batch_labelled + batch_unlabelled
        total_loss = (batch_labelled / batch_total) * labelled_loss + (batch_unlabelled / batch_total) * unlabelled_loss
        # total_loss = unlabelled_loss
        targets = torch.argmax(l_arc_scores, dim=2)

        num_correct = 0
        total = 0

        num_correct += torch.count_nonzero((targets == parents) * (parents != -1))
        total += torch.count_nonzero((parents == parents) * (parents != -1))
        
        """
        arc_scores, lab_scores = self(val_batch)

        arc_loss = self.arc_loss(arc_scores,targets)
        lab_loss = self.lab_loss(lab_scores, targets, labels)

        total_loss = arc_loss + lab_loss
        num_correct = 0
        total = 0
        
        parents = torch.argmax(arc_scores, dim=2)
        num_correct += torch.count_nonzero((parents == targets) * (targets != -1))
        total += torch.count_nonzero((targets == targets)* (targets != -1))
        """

        self.log('validation_loss', total_loss, on_step=False, on_epoch=True, logger=True)
        self.log('validation_labelled_loss', labelled_loss.detach(), on_step=False, on_epoch=True, logger=True)
        self.log('validation_unlabelled_loss', unlabelled_loss.detach(), on_step=False, on_epoch=True, logger=True)
        
        return {'loss': total_loss, 'correct': num_correct, 'total': total, 'labelled_loss':labelled_loss.detach(), 'unlabelled_loss': unlabelled_loss.detach()}

    def validation_epoch_end(self, preds):
        correct = 0
        total = 0
        loss = 0
        arc_loss = 0.0
        lab_loss = 0.0
        labelled_loss = 0.0
        unlabelled_loss = 0.0
        for pred in preds:
            correct += pred['correct']
            total += pred['total']
            loss += pred['loss']
            labelled_loss += pred['labelled_loss']
            unlabelled_loss += pred['unlabelled_loss']
            # arc_loss += pred['arc_loss'] / len(preds)
            # lab_loss += pred['lab_loss'] / len(preds)

        self.log('validation_loss', loss, on_step=False, on_epoch=True, logger=True)
        self.log("validation_accuracy",correct/total, on_epoch=True, logger=True)

        print('\nAccuracy on validation set: {:3.3f} | Loss : {:3.3f} | Labelled loss: {:3.3f} | Unlabelled loss: {:3.3f}'.format(correct/total, loss/len(preds), labelled_loss.detach()/len(preds), unlabelled_loss.detach()/len(preds)))
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

        #lab_loss = self.lab_loss(lab_scores, targets, labels)

        total_loss = arc_loss #+ lab_loss
        num_correct = 0
        total = 0

        num_correct += torch.count_nonzero((trees == targets) * (targets != -1))
        total += torch.count_nonzero((targets == targets)* (targets != -1))

        self.log('test_loss', total_loss.detach(), on_step=False, on_epoch=True, logger=True)
        self.log('test_arc_loss', arc_loss.detach(), on_step=False, on_epoch=True, logger=True)
        #self.log('test_lab_loss', lab_loss.detach(), on_step=False, on_epoch=True, logger=True)

        return {'loss': total_loss, 'correct': num_correct, 'total': total, 'arc_loss':arc_loss.detach()}

    
    def test_epoch_end(self, preds):
        correct = 0
        total = 0
        loss = 0.0
        for pred in preds:
            correct += pred['correct']
            total += pred['total']
            loss += pred['loss']
        
        print('\nAccuracy on test set: {:3.3f} | Loss : {:3.3f}\n'.format(correct/total, loss/len(preds)))
        
        return {'accuracy': correct/total, 'loss': loss}

    def GE_loss(self, scores, features):
        X = torch.einsum('bij,bijf->bf', scores, features)
        Y = X - self.prior
        return 0.5 * (Y.unsqueeze(-2) @ Y.unsqueeze(-1)).squeeze(-1).squeeze(-1)

    def entropy_loss(self, marginals, scores):
        mask = scores == inf
        masked = scores.masked_fill(mask, 0.0)
        X = torch.einsum('bij,bij->b', marginals, masked).sum()
        return X

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

    def arc_loss(self, S_arc, heads):
        """Compute the loss for the arc predictions."""
        #S_arc = S_arc.transpose(-1, -2)                      # [batch, sent_len, sent_len]
        S_arc = S_arc.contiguous().view(-1, S_arc.size(-1))  # [batch*sent_len, sent_len]
        heads = heads.view(-1)                               # [batch*sent_len]
        return self.loss(S_arc, heads)

    def apply_log_softmax(self, scores, lengths):
        lengths = lengths - torch.ones(len(lengths))

        batch, maxlen, _ = scores.shape
        pads = torch.arange(maxlen) >= lengths.unsqueeze(1)
        mask = pads.unsqueeze(1) | pads.unsqueeze(2)

        # set the scores of arcs to padding tokens to a large negative number
        scores.masked_fill_(mask, -1e9)
        aux = self.log_softmax(scores)
        mask = aux <= -1e6
        # aux.masked_fill_(mask, -inf)
        return aux.masked_fill(mask, -inf)

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
        
