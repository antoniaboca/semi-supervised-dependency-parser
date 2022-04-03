from torch.nn.modules import linear
from dependency_parsers.data.processor import PAD_VALUE, unlabelled_padder
from dependency_parsers.nn.layers import Biaffine, MLP
from dependency_parsers.nn.losses import arc_loss, edmonds_arc_loss, GE_loss
from dependency_parsers.nn.transform import score_to_diagonal, apply_log_softmax, feature_to_diagonal
import torchmetrics

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.functional as F

from torch.nn.utils.rnn import pad_sequence, pad_packed_sequence, pack_padded_sequence
from torch.nn import Linear, LSTM

from numpy import inf

from torch_struct import NonProjectiveDependencyCRF

from stanza.models.common.chuliu_edmonds import chuliu_edmonds_one_root

import pytorch_lightning as pl

class LitSemiSupervisedLSTM(pl.LightningModule):
    def __init__(self, embeddings, f_star, embedding_dim, hidden_dim, num_layers, 
                lstm_dropout, linear_dropout, arc_dim, lab_dim, num_labels, lr, loss_arg, cle_arg, ge_only, vocabulary, order, labelled_ratio, tag_type):
        super().__init__()
        self.save_hyperparameters()

        self.hidden_dim = hidden_dim
        self.arc_dim = arc_dim
        self.lab_dim = lab_dim
        self.lr = lr
        self.cle = cle_arg
        self.prior = f_star
        self.ge_only = ge_only
        self.vocabulary = vocabulary
        self.order = order
        self.labelled_ratio = labelled_ratio
        self.tag_type = tag_type

        self.lstm = LSTM(
            input_size=embedding_dim, 
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=lstm_dropout,
            bidirectional=True
        )

        self.word_embedding = nn.Embedding.from_pretrained(torch.tensor(embeddings), 
                                                            padding_idx=0)

        # arc and label MLP layers
        self.linear_dropout = linear_dropout
        
        self.MLP_arc = MLP(self.hidden_dim * 2, self.linear_dropout, self.arc_dim)
        self.MLP_lab = MLP(self.hidden_dim * 2, self.linear_dropout, self.lab_dim)

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
    
    def assert_features(self, batch):
        batch = batch['unlabelled']
        oracle_edges = 0
        total_edges = 0

        for idx in range(len(batch['sentence'])):
            length = batch['lengths'][idx]
            features = batch['features'][idx]
            pos = batch[self.tag_type][idx]
            parents = batch['parents'][idx]
            for i in range(length):
                if i > 0:
                    parent = parents[ i ]
                    x1 = int(pos[i])
                    x2 = int(pos[parent])
                    if (x2, x1) in self.order:
                        oracle_edges += 1
                    total_edges += 1

                for j in range(length):
                    if torch.any(features[i][j] == 1.0):
                        x1 = int(pos[i])
                        x2 = int(pos[j])
                        assert (x1, x2) in self.order
                        
                        ord = self.order[(x1, x2)]
                        assert features[i][j][ord] == 1.0
                        assert torch.all(features[i][j][:ord] == 0.0)
                        assert torch.all(features[i][j][(ord+1):] == 0.0)
                    else:
                        assert torch.all(features[i][j] == 0.0)
        
        return (oracle_edges / total_edges)

    def forward(self, x):
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
        edge_ratio = self.assert_features(batch)

        unlabelled = batch['unlabelled']
        features = unlabelled['features']

        # COMPUTE LOSS FOR UNLABELLED DATA
        arc_scores, lab_scores = self(unlabelled)
        batch_unlabelled, maxlen_unlabelled, _ = arc_scores.shape
        lengths = unlabelled['lengths']
        
        potentials = score_to_diagonal(arc_scores)
        log_potentials = apply_log_softmax(potentials, lengths)
        dist = NonProjectiveDependencyCRF(log_potentials, lengths - torch.ones(len(lengths)))
    
        assert not torch.isnan(dist.partition).any()
        assert not torch.isinf(dist.partition).any()
        assert not torch.isinf(dist.marginals).any()
        assert not torch.isnan(dist.marginals).any()

        diag_features = feature_to_diagonal(features.float())
        unlabelled_loss = GE_loss(dist.marginals, diag_features, self.prior).mean()

        self.log('unlabelled_loss', unlabelled_loss.detach(), on_step=True, on_epoch=True, logger=True)

        if self.ge_only:
            self.log('training_loss', unlabelled_loss.detach(), on_step=True, on_epoch=True, logger=True)
            return {
                'loss': unlabelled_loss,
                'edge_ratio': edge_ratio,
            }

        # COMPUTE LOSS FOR LABELLED DATA
        labelled = batch['labelled']
        l_arc_scores, l_lab_scores = self(labelled)
        batch_labelled, maxlen_labelled, _ = l_arc_scores.shape
        lengths = labelled['lengths']
        parents = labelled['parents']

        labelled_loss = arc_loss(l_arc_scores, parents, self.loss)
        total_loss = self.labelled_ratio * labelled_loss + (1 - self.labelled_ratio) * unlabelled_loss

        targets = torch.argmax(l_arc_scores, dim=2)

        num_correct = 0
        total = 0

        num_correct += torch.count_nonzero((targets == parents) * (parents != -1))
        total += torch.count_nonzero((parents == parents) * (parents != -1))
        
        self.log('training_loss', total_loss.detach(), on_step=True, on_epoch=True, logger=True)
        
        return {'loss': total_loss, 
                'correct': num_correct, 
                'total': total, 
                'labelled_loss': labelled_loss.detach(),
                'unlabelled_loss': unlabelled_loss.detach(),
        }
        
    def training_epoch_end(self, outputs):
        correct = 0
        total = 0
        loss = 0.0
        labelled = 0.0
        unlabelled = 0.0

        ratio = 0.0
        for output in outputs:
            loss += output['loss'] / len(outputs)
            if self.ge_only:
                ratio += output['edge_ratio']

            if not self.ge_only:
                labelled += output['labelled_loss'] / len(outputs)
                unlabelled += output['unlabelled_loss'] / len(outputs)
                correct += output['correct']
                total += output['total']
            
        if not self.ge_only:
            self.log('training_accuracy', correct/total, on_epoch=True, on_step=False, logger=True)
            print('\nAccuracy on labelled data: {:3.3f} | GE Loss: {:3.3f} | Labelled loss: {:3.3f}'.format(correct/total, unlabelled, labelled))
        print('\nOracle target edges found: {:3.3f}% \n'.format(ratio / len(outputs)))

    def validation_step(self, val_batch, batch_idx):
        targets = val_batch['parents']
        labels = val_batch['labels']
        lengths = val_batch['lengths']
        features = val_batch['features']

        batch, maxlen = targets.shape
        arc_scores, lab_scores = self(val_batch)

        potentials = score_to_diagonal(arc_scores)
        log_potentials = apply_log_softmax(potentials, lengths)
        dist = NonProjectiveDependencyCRF(log_potentials, lengths - torch.ones(len(lengths)))

        assert not torch.isnan(dist.partition).any()
        assert not torch.isinf(dist.partition).any()
        assert not torch.isinf(dist.marginals).any()
        assert not torch.isnan(dist.marginals).any()

        diag_features = feature_to_diagonal(features.float())
        unlabelled_loss = GE_loss(dist.marginals, diag_features, self.prior).mean()
        labelled_loss = arc_loss(arc_scores, targets, self.loss)
        # lab_loss = self.lab_loss(lab_scores, targets, labels)

        total_loss = self.labelled_ratio * labelled_loss + (1.0 - self.labelled_ratio) * unlabelled_loss
        
        if self.ge_only:
            total_loss = unlabelled_loss
        
        num_correct = 0
        total = 0
        
        parents = torch.argmax(arc_scores, dim=2)
        num_correct += torch.count_nonzero((parents == targets) * (targets != -1))
        total += torch.count_nonzero((targets == targets)* (targets != -1))

        self.log('validation_loss', total_loss.detach(), on_step=False, on_epoch=True, logger=True)
        self.log('validation_labelled_loss', labelled_loss.detach(), on_step=False, on_epoch=True, logger=True)

        return {'loss': total_loss, 'correct': num_correct, 'total': total, 'arc_loss': labelled_loss.detach()}

    def validation_epoch_end(self, preds):
        correct = 0
        total = 0
        loss = 0
        arc_loss = 0.0
        for pred in preds:
            correct += pred['correct']
            total += pred['total']
            loss += pred['loss']
            arc_loss += pred['arc_loss'] / len(preds)

        self.log("validation_accuracy",correct/total, on_epoch=True, logger=True)

        print('\nAccuracy on validation set: {:3.3f} | Loss : {:3.3f} | Arc loss: {:3.3f} '.format(correct/total, loss/len(preds), arc_loss.detach()))
        return {'accuracy': correct / total, 'loss': loss/len(preds)}

    def test_step(self, test_batch, batch_idx):
        targets = test_batch['parents']
        labels = test_batch['labels']
        lengths = test_batch['lengths']

        batch, maxlen = targets.shape
        arc_scores, lab_scores = self(test_batch)
        
        if self.cle:
            trees, labelled_loss = edmonds_arc_loss(arc_scores, lengths, targets, self.loss)
        else:
            trees = torch.argmax(arc_scores, dim=2)
            labelled_loss = arc_loss(arc_scores, targets, self.loss)

        total_loss = labelled_loss
        num_correct = 0
        total = 0

        num_correct += torch.count_nonzero((trees == targets) * (targets != -1))
        total += torch.count_nonzero((targets == targets)* (targets != -1))

        self.log('test_loss', total_loss.detach(), on_step=False, on_epoch=True, logger=True)
        self.log('test_arc_loss', labelled_loss.detach(), on_step=False, on_epoch=True, logger=True)

        return {'loss': total_loss, 'correct': num_correct, 'total': total, 'arc_loss':labelled_loss.detach()}

    
    def test_epoch_end(self, preds):
        return self.validation_epoch_end(preds)

        
