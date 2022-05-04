import torch
from numpy import inf 

import torch.nn.functional as F

def apply_log_softmax(scores, lengths):
    """
        Applies the LogSoftmax function to a set of scores while ensuring
        numerical safety for padding scores.
    """
    lengths = lengths - torch.ones(len(lengths))

    batch, maxlen, _ = scores.shape
    pads = torch.arange(maxlen) >= lengths.unsqueeze(1)
    mask = pads.unsqueeze(1) | pads.unsqueeze(2)

    # set the scores of arcs to padding tokens to a large negative number
    scores.masked_fill_(mask, -1e9)
        
    aux1 = F.log_softmax(torch.reshape(scores, (batch, maxlen * maxlen)), dim=-1)
    aux2 = torch.reshape(aux1, (batch, maxlen, maxlen)) 
    mask = aux2 <= -1e6
    return aux2.masked_fill(mask, -inf)

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