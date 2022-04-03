import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import Linear

class Biaffine(nn.Module):
    def __init__(self, arc_dim, output_dim, from_pretrained=None):
        super().__init__()
        if from_pretrained is None:
            self.W = nn.Parameter(torch.Tensor(output_dim, arc_dim, arc_dim))
            self.reset_parameters()
        
        else:
            if type(from_pretrained) is not torch.Tensor:
                from_pretrained = torch.tensor(from_pretrained)
            assert from_pretrained.shape == (output_dim, arc_dim, arc_dim)
            self.W = nn.Parameter(from_pretrained)

    def forward(self, head, dep):
        head = head.unsqueeze(1)
        dep = dep.unsqueeze(1)
        scores = head @ self.W @ dep.transpose(-1,-2)
        return scores.squeeze(1)

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W)

class MLP(nn.Module):
    def __init__(self, hidden_dim, linear_dropout, out_dim):
        super().__init__()

        # Dropout layer
        self.dropout = nn.Dropout(linear_dropout)

        # Arc Linear Layer
        self.linear_h = Linear(hidden_dim, out_dim)  # this is your g
        self.linear_d = Linear(hidden_dim, out_dim) # this is your f

        self.score_h = Linear(out_dim, 1)
        self.score_d = Linear(out_dim, 1)

    def forward(self, x):
        h_out = self.dropout(F.relu(self.linear_h(x)))
        d_out = self.dropout(F.relu(self.linear_d(x)))

        h_score = self.score_h(h_out)
        d_score = self.score_d(d_out)
        return h_out, d_out, h_score, d_score
    
    
    