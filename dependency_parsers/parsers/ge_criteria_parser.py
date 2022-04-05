import torchmetrics

from numpy import inf

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence, pad_packed_sequence, pack_padded_sequence
from torch.nn import Linear, LSTM

from dependency_parsers.nn.layers import BiaffineLSTM
from dependency_parsers.nn.losses import arc_loss, edmonds_arc_loss, GE_loss
from dependency_parsers.nn.transform import score_to_diagonal, apply_log_softmax, feature_to_diagonal

from torch_struct import NonProjectiveDependencyCRF
from stanza.models.common.chuliu_edmonds import chuliu_edmonds_one_root
import pytorch_lightning as pl

class LitSemiSupervisedLSTM(pl.LightningModule):
    """PyTorch Lightning module that trains the Biaffine parser using the GE criteria technique.

    Attributes:
        lr (float): Learning rate of the optimiser.
        cle (boolean): Whether the final prediction for the dependency tree is computed using 
            Edmonds' Algorithm.
        prior (torch.Tensor): The prior distribution of the top 20 POS tag edges
        ge_only (boolean): Whether the parser should train using a split of labelled and unlabelled
            data or using only the GE criteria on unlabelled data.
        vocabulary (allennlp.data.Vocabulary): the vocabulary of the training set
        order (dict): A dictionary having the top 20 edges as keys and their order in the top as values.
        labelled_ratio (float): The ratio used in calculating the combined loss between supervised and 
            unsupervised scores.
        tag_type (str): Either 'xpos' or 'upos', the type of tag to be used in training.
        model (nn.Module): the Biaffine Parser to train.
        loss (nn.CrossEntropyLoss): the PyTorch implementation of the Cross Entropy loss function
    """
    def __init__(self, embeddings, f_star, vocabulary, order, args):
        """The Constructor method of the GE Criteria Parser.

        Args:
            embeddings (numpy.ndarray): Embeddings to be used in the Encoder
            f_star (torch.Tensor): The prior distribution of the top 20 POS tag edges
            vocabulary (allennlp.data.Vocabulary): the vocabulary of the training set
            order (dict): A dictionary having the top 20 edges as keys and their order 
                in the top as values.
            args (object): Object containing arguments used to set up the hyperparameters
                of the network.
        """
        super().__init__()
        self.save_hyperparameters()

        self.lr = args.lr
        self.cle = args.cle
        self.prior = f_star
        self.ge_only = args.ge_only
        self.vocabulary = vocabulary
        self.order = order
        self.labelled_ratio = args.labelled_ratio
        self.tag_type = args.tag_type

        self.model = BiaffineLSTM(embeddings, args)

        self.loss = nn.CrossEntropyLoss(ignore_index=-1)
    
    def configure_optimizers(self):
        """Pytorch Lightning method that sets up the optimisation algorithm.

        Returns:
            torch.optim.Optimizer: Optimiser object for PyTorch.
        """
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
    
    def assert_features(self, batch):
        """Function that asserts the feature function of each sentence has the correct format.

        Args:
            batch (dict): A dictionary representing the input to the network. The dictionary contains:
                sentence indexes, UPOS tag indexes, XPOS tag indexes, parent indexes, label indexes,
                the length of the sentence

        Returns:
            float: Ratio indicating how many edges in the labelled trees are covered by the top 20 edges.
        """
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
        """Defines the computation performed at every call of the model.
        Args:
            x (dict): A dictionary representing the input to the network. The dictionary contains:
                sentence indexes, UPOS tag indexes, XPOS tag indexes, parent indexes, label indexes,
                the length of the sentence

        Returns:
            tuple: The arc and label scores predicted by the model.
        """
        return self.model(x)
        
    def training_step(self, batch, batch_idx):
        """A PyTorch Lightning method that gets called on each batch used in a training epoch.

        Args:
            train_batch (dict): A dictionary representing the input to the network. The dictionary contains:
                sentence indexes, UPOS tag indexes, XPOS tag indexes, parent indexes, label indexes,
                the length of the sentence
            batch_idx (int): The index of the current batch in the current epoch.

        Returns:
            dict: A dictionary that contains the computed loss for the current batch, the arc loss, the label
                loss, the number of correctly identified edges in the trees and the total number of edges 
                in the trees.
        """
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
                'edge_ratio': edge_ratio
        }
        
    def training_epoch_end(self, outputs):
        """PyTorch Lightning method that gets called at the end of each training epoch.

        Args:
            outputs (Iterable): An iterable object containing the outputs of each training_step
                call.
        """
        correct = 0
        total = 0
        loss = 0.0
        labelled = 0.0
        unlabelled = 0.0

        ratio = 0.0
        for output in outputs:
            loss += output['loss'] / len(outputs)
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
        """PyTorch Lightning method that gets called for each batch in a validation epoch.

        Args:
            val_batch (dict): A dictionary representing the input to the network. The dictionary contains:
                sentence indexes, UPOS tag indexes, XPOS tag indexes, parent indexes, label indexes,
                the length of the sentence
            batch_idx (int): The index of the current batch in the current epoch.

        Returns:
            dict: Dictionary of statistics for the current batch.
        """
        targets = val_batch['parents']
        labels = val_batch['labels']
        lengths = val_batch['lengths']
        features = val_batch['features']

        batch, maxlen = targets.shape
        arc_scores, lab_scores = self(val_batch)


        # COMPUTE THE UNLABELLED LOSS
        potentials = score_to_diagonal(arc_scores)
        log_potentials = apply_log_softmax(potentials, lengths)
        dist = NonProjectiveDependencyCRF(log_potentials, lengths - torch.ones(len(lengths)))

        assert not torch.isnan(dist.partition).any()
        assert not torch.isinf(dist.partition).any()
        assert not torch.isinf(dist.marginals).any()
        assert not torch.isnan(dist.marginals).any()

        diag_features = feature_to_diagonal(features.float())
        unlabelled_loss = GE_loss(dist.marginals, diag_features, self.prior).mean()
        
        # COMPUTE THE LABELLED LOSS
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
        """PyTorch Lightning method that gets called at the end of a validation epoch.

        Args:
            preds (Iterable): An Iterable object with the statistics of each validation_step call.

        Returns:
            dict: Dictionary of overall statistics for the current epoch to be printed and logged.
        """
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
        """PyTorch Lightning method that gets called for each batch in the testing set.

        Args:
            test_batch (dict): A dictionary representing the input to the network. The dictionary contains:
                sentence indexes, UPOS tag indexes, XPOS tag indexes, parent indexes, label indexes,
                the length of the sentence
            batch_idx (int): The index of the current batch in the current epoch.

        Returns:
            dict: Dictionary of statistics for the current batch.
        """
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
        """PyTorch Lightning method that gets called at the end of a validation epoch.

        Args:
            preds (Iterable): An Iterable object with the statistics of each validation_step call.

        Returns:
            dict: Dictionary of overall statistics for the current epoch to be printed and logged.
        """
        return self.validation_epoch_end(preds)

        
