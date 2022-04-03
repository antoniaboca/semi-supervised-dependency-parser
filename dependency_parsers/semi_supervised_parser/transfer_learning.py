from torch._C import set_flush_denormal
from torch.nn.modules import linear
import torchmetrics

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.functional as F

from torch_struct import NonProjectiveDependencyCRF

from stanza.models.common.chuliu_edmonds import chuliu_edmonds_one_root

import pytorch_lightning as pl

from dependency_parsers.biaffine_parser.biaffine_lstm import LitLSTM
from dependency_parsers.nn.transform import score_to_diagonal, apply_log_softmax, feature_to_diagonal
from dependency_parsers.nn.losses import GE_loss, arc_loss, lab_loss, edmonds_arc_loss

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

        # COMPUTE LOSS FOR LABELLED DATA

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

        arcs = arc_loss(arc_scores,targets, self.loss)
        labs = lab_loss(lab_scores, targets, labels, self.loss)

        total_loss = arcs + labs
        num_correct = 0
        total = 0
        
        parents = torch.argmax(arc_scores, dim=2)
        num_correct += torch.count_nonzero((parents == targets) * (targets != -1))
        total += torch.count_nonzero((targets == targets)* (targets != -1))

        self.log('validation_loss', total_loss.detach(), on_step=False, on_epoch=True, logger=True)
        self.log('validation_arc_loss', arcs.detach(), on_step=False, on_epoch=True, logger=True)
        self.log('validation_lab_loss', labs.detach(), on_step=False, on_epoch=True, logger=True)

        return {'loss': total_loss, 'correct': num_correct, 'total': total, 'arc_loss':arcs.detach(), 'lab_loss': labs.detach()}

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
            trees, arcs = edmonds_arc_loss(arc_scores, lengths, targets)
        else:
            trees = torch.argmax(arc_scores, dim=2)
            arcs = arc_loss(arc_scores, targets, self.loss)

        labs = lab_loss(lab_scores, targets, labels, self.loss)

        total_loss = arcs + labs
        num_correct = 0
        total = 0

        num_correct += torch.count_nonzero((trees == targets) * (targets != -1))
        total += torch.count_nonzero((targets == targets)* (targets != -1))

        self.log('test_loss', total_loss.detach(), on_step=False, on_epoch=True, logger=True)
        self.log('test_arc_loss', arcs.detach(), on_step=False, on_epoch=True, logger=True)
        self.log('test_lab_loss', labs.detach(), on_step=False, on_epoch=True, logger=True)

        return {'loss': total_loss, 'correct': num_correct, 'total': total, 'arc_loss':arcs.detach(), 'lab_loss': labs.detach()}

    
    def test_epoch_end(self, preds):
        return self.validation_epoch_end(preds)