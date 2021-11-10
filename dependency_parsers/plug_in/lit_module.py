import torch
import pytorch_lightning as pl
from torch import nn
import torch.optim as optim

class LitModel(pl.LightningModule):
    def __init__(self, model):
        super(LitModel, self).__init__()
        self.model = model

        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform(p)
    
    def forward(self, x):
        return self.model(x)
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=0.05)
        return optimizer

    def training_step(self, batch, batch_idx):
        targets = batch['parents']
        targets[:, 0] = -100
        parent_scores = self(batch)

        total_loss = self.model.arc_loss(parent_scores, targets)
        acc = self.arc_accuracy(parent_scores, targets)

        return {'loss': total_loss, 'accuracy': acc}
    
    def arc_accuracy(self, S_arc, heads, eps=1e-10):
        """Accuracy of the arc predictions based on gready head prediction."""
        pred = torch.argmax(S_arc, dim=-2)
        mask = (heads != -100).float()
        accuracy = torch.sum((pred == heads).float() * mask, dim=-1) / (torch.sum(mask, dim=-1) + eps)
        return torch.mean(accuracy).item()


    def training_epoch_end(self, outputs):
        acc = 0
        for output in outputs:
            acc += output['accuracy']
        acc /= len(outputs)
        print('\nAccuracy after epoch end: {:3.2f}'.format(acc))
