import torch
from numpy import inf 

import torch.nn.functional as F

# numerically safe log softmax
def apply_log_softmax(scores, lengths):
    lengths = lengths - torch.ones(len(lengths))

    batch, maxlen, _ = scores.shape
    pads = torch.arange(maxlen) >= lengths.unsqueeze(1)
    mask = pads.unsqueeze(1) | pads.unsqueeze(2)

    # set the scores of arcs to padding tokens to a large negative number
    scores.masked_fill_(mask, -1e9)
        
    aux = F.log_softmax(scores, dim=-1)
    mask = aux <= -1e6
    return aux.masked_fill(mask, -inf)

def feature_to_diagonal(features):
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

def score_to_diagonal(scores):
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