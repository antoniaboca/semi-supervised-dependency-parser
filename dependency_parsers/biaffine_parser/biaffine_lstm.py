
import torchmetrics

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.functional as F

from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from torch.nn import LSTM

from dependency_parsers.nn.layers import Biaffine, MLP
from dependency_parsers.nn.losses import lab_loss, arc_loss, edmonds_arc_loss


import pytorch_lightning as pl

class LitLSTM(pl.LightningModule):
    def __init__(self, embeddings, embedding_dim, hidden_dim, num_layers, 
                lstm_dropout, linear_dropout, arc_dim, lab_dim, num_labels, lr, loss_arg, cle_arg):
        super().__init__()

        self.save_hyperparameters()
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

        self.linear_dropout = linear_dropout

        # Linear layers 

        self.MLP_arc = MLP(self.hidden_dim * 2, self.linear_dropout, self.arc_dim)
        self.MLP_lab = MLP(self.hidden_dim * 2, self.linear_dropout, self.lab_dim)
        
        # biaffine layers
        self.arc_biaffine = Biaffine(arc_dim, 1)
        self.lab_biaffine = Biaffine(lab_dim, num_labels)

        self.arc_loss = arc_loss
        self.lab_loss = lab_loss
        self.edmonds_arc_loss = edmonds_arc_loss

        self.loss = nn.CrossEntropyLoss(ignore_index=-1)
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
    
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
        
    def training_step(self, train_batch, batch_idx):
        parents = train_batch['parents']
        labels = train_batch['labels']

        arc_scores, lab_scores = self(train_batch)
        batch, maxlen, _ = arc_scores.shape
        lengths = train_batch['lengths']

        arc_loss = self.arc_loss(arc_scores, parents, self.loss)
        lab_loss = self.lab_loss(lab_scores, parents, labels, self.loss)

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
        
        self.log('training_accuracy', correct/total, on_epoch=True, on_step=False, logger=True)
        print('\nAccuracy after epoch end: {:3.3f} | Arc loss: {:3.3f} | Lab loss: {:3.3f}'.format(correct/total, arc_loss.detach(), lab_loss.detach()))

    def validation_step(self, val_batch, batch_idx):
        targets = val_batch['parents']
        labels = val_batch['labels']
        
        batch, maxlen = targets.shape
        arc_scores, lab_scores = self(val_batch)

        arc_loss = self.arc_loss(arc_scores, targets, self.loss)
        lab_loss = self.lab_loss(lab_scores, targets, labels, self.loss)

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
            trees, arc_loss = self.edmonds_arc_loss(arc_scores, lengths, targets, self.loss)
        else:
            trees = torch.argmax(arc_scores, dim=2)
            arc_loss = self.arc_loss(arc_scores, targets, self.loss)

        lab_loss = self.lab_loss(lab_scores, targets, labels, self.loss)

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


 

