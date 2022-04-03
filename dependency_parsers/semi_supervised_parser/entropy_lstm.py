import torchmetrics

from cmath import inf

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from dependency_parsers.biaffine_parser.biaffine_lstm import LitSupervisedLSTM
from dependency_parsers.nn.layers import BiaffineLSTM
from dependency_parsers.nn.losses import entropy_loss, arc_loss, edmonds_arc_loss
from dependency_parsers.nn.transform import feature_to_diagonal, score_to_diagonal, apply_log_softmax

from torch.nn.utils.rnn import pad_sequence, pad_packed_sequence, pack_padded_sequence
from torch.nn import Linear, LSTM

from torch_struct import NonProjectiveDependencyCRF
import pytorch_lightning as pl

class LitEntropyLSTM(pl.LightningModule):
    def __init__(self, embeddings, args):
        super().__init__()

        self.loss = nn.CrossEntropyLoss(ignore_index=-1)

        self.transfer = args.transfer
        self.log_softmax = nn.LogSoftmax(dim=-1)
        self.ge_only = args.ge_only

        if args.transfer is True:
            self.lr = args.lr
            self.cle = args.cle
            self.model = LitSupervisedLSTM.load_from_checkpoint('my_logs/default/version_73/checkpoints/epoch=15-step=1007.ckpt', 
                            hparams_file='my_logs/default/version_73/hparams.yaml')
        else:
            self.model = BiaffineLSTM(embeddings, args)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
    
    def forward(self, x):
        return self.model(x)
        
    def training_step(self, batch, batch_idx):
        unlabelled = batch['unlabelled']
        features = unlabelled['features']

        # COMPUTE LOSS FOR UNLABELLED DATA
        arc_scores, lab_scores = self(unlabelled)
        batch_unlabelled, maxlen_unlabelled, _ = arc_scores.shape
        lengths = unlabelled['lengths']
        
        potentials = score_to_diagonal(arc_scores)
        log_potentials = apply_log_softmax(potentials, lengths)

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
        unlabelled_loss = entropy_loss(dist.marginals, torch.clone(-log_potentials))

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

        labelled_loss = arc_loss(l_arc_scores, parents, self.loss)
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
        
        potentials = score_to_diagonal(arc_scores)
        log_potentials = apply_log_softmax(potentials, lengths)

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
        unlabelled_loss = entropy_loss(dist.marginals, torch.clone(-log_potentials))

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

        labelled_loss = arc_loss(l_arc_scores, parents, self.loss)
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
            trees, labelled_loss = edmonds_arc_loss(arc_scores, lengths, targets)
        else:
            trees = torch.argmax(arc_scores, dim=2)
            labelled_loss = arc_loss(arc_scores, targets, self.loss)

        total_loss = labelled_loss
        num_correct = 0
        total = 0

        num_correct += torch.count_nonzero((trees == targets) * (targets != -1))
        total += torch.count_nonzero((targets == targets)* (targets != -1))

        self.log('test_loss', total_loss.detach(), on_step=False, on_epoch=True, logger=True)
        self.log('test_arc_loss', arc_loss.detach(), on_step=False, on_epoch=True, logger=True)

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
