import torch
import torch.nn as nn
import torch.nn.functional as F

from numpy import inf
from torch.nn.utils.rnn import pad_sequence
from stanza.models.common.chuliu_edmonds import chuliu_edmonds_one_root
from torch_struct import NonProjectiveDependencyCRF

def lab_loss(S_lab, parents, labels, loss):
    """Compute the loss for the label predictions on the gold arcs (heads)."""
    heads = torch.clone(parents)
    heads[heads == -1] = 0
    heads = heads.unsqueeze(1).unsqueeze(2)              # [batch, 1, 1, sent_len]
    heads = heads.expand(-1, S_lab.size(1), -1, -1)      # [batch, n_labels, 1, sent_len]
    S_lab = torch.gather(S_lab, 2, heads).squeeze(2)     # [batch, n_labels, sent_len]
    S_lab = S_lab.transpose(-1, -2)                      # [batch, sent_len, n_labels]
    S_lab = S_lab.contiguous().view(-1, S_lab.size(-1))  # [batch*sent_len, n_labels]
    labels = labels.view(-1)      
    return loss(S_lab, labels)

def arc_loss(S_arc, heads, loss):
    """Compute the loss for the arc predictions."""
    #S_arc = S_arc.transpose(-1, -2)                      # [batch, sent_len, sent_len]
    S_arc = S_arc.contiguous().view(-1, S_arc.size(-1))  # [batch*sent_len, sent_len]
    heads = heads.view(-1)                               # [batch*sent_len]
    return loss(S_arc, heads)

def edmonds_arc_loss(S_arc, lengths, heads, loss):
    """Compute the loss for the arc predictions using the Edmonds algorithm."""
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

    return batched, arc_loss(S_arc, heads, loss)

def log_partition(scores, length):
    """Computes Z, the partition functions, given a batch of scores for some dependency graphs."""
    batch, slen, slen_ = scores.shape
    assert slen == slen_

    pads = torch.arange(slen) >= length.unsqueeze(1)
    mask = pads.unsqueeze(1) | pads.unsqueeze(2)

    # set the scores of arcs to padding tokens to a large negative number
    scores.masked_fill_(mask, -1e9)
    
    # set the scores of arcs incoming to root
    scores[:, :, 0] = -1e9
    
    # set the scores of self loops 
    scores.masked_fill_(torch.eye(slen).bool(), -1e9)

    max_score, _ = scores.reshape(batch, -1).max(dim=1)
    weights = (scores - max_score.reshape(batch, 1, 1)).exp() + 1e-8

    weights[:, 0].masked_fill_(pads, 1.)
    w = weights.masked_fill(torch.eye(slen).bool(), 0)

    # Create the Laplacian matrix
    laplacian = -weights
    laplacian.masked_fill_(torch.eye(slen).bool(), 0)
    laplacian += torch.diag_embed(w.sum(dim=1))

    # Compute log partition with MTT
    # 
    # The MTT states that the log partition is equal to the determinant of the matrix 
    # obtained by removing the first row and column from the Laplacian matrix for the weights

    log_part = laplacian[:, 1:, 1:].logdet()
    log_part = log_part + (length.float() - 1) * max_score

    return log_part

def MTT_loss_1(parent_scores, lengths, targets):
    """Computes the MTT loss, that uses the Z function, using a technique for numerical safety."""
    batch, maxlen, _ = parent_scores.shape

    # create masks to work with 
    pads = torch.arange(maxlen) >= lengths.unsqueeze(1)
    mask = pads.unsqueeze(1) | pads.unsqueeze(2)
    rows = pads.unsqueeze(-1).expand((batch, maxlen, maxlen))

    # set the scores of arcs to padding tokens to a large negative number
    parent_scores.masked_fill_(mask, -1e9)
    # set the scores of arcs incoming to root
    parent_scores[:, :, 0] = -1e9
    # set the scores of self loops 
    parent_scores.masked_fill_(torch.eye(maxlen).bool(), -1e9)

    # normalize scores usign softmax
    _S = F.log_softmax(parent_scores, dim=-1)
    S = torch.clone(_S)

    # compute the log partition of the graph using MTT
    Z = log_partition(S, lengths)

    S = torch.clone(_S)
    # set the scores of arcs to padding tokens to 0
    S.masked_fill_(mask, 0)
    # set the scores of arcs incoming to root
    S[:, :, 0] = 0
    # set the scores of self loops n
    S.masked_fill_(torch.eye(maxlen).bool(), 0)

    assert maxlen == targets.size(1)

    S = S.reshape((batch * maxlen, maxlen))
    valid = torch.reshape(targets, (batch * maxlen,))

    masker = valid == -1
    offset = torch.arange(start=0, end=maxlen * batch, step=maxlen).unsqueeze(-1).expand(-1, maxlen).reshape((batch * maxlen,))
    indexer = torch.arange(maxlen).repeat(batch, 1).reshape((batch*maxlen,))
    assert valid.shape == indexer.shape
        
    valid[masker] = 0
    valid = valid + offset

    # get the sum of edges of each target tree in the batch
    sums = S[valid, indexer].reshape(batch, maxlen)

    # compute the negative log likelihood of each tree
    P = - (sums.sum(dim=-1) - Z)
    try:
        assert P.mean() > 0
    except AssertionError:
        import ipdb
        ipdb.post_mortem()
    return P.mean()

def MTT_loss_2(scores, targets, lengths, Z):
    """Computes the MTT loss, using the Z function, using a different technique for numerical safety."""
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

def MTT_loss_no_Z(arc_scores, lengths, targets):
    """Computes the MTT loss ignoring Z (that is supposed to normalise the scores)."""
    batch, maxlen, _ = arc_scores.shape
    S = NonProjectiveDependencyCRF(arc_scores, lengths)

    S = S.reshape((batch * maxlen, maxlen))
    valid = torch.reshape(targets, (batch * maxlen,))

    offset = torch.arange(start=0, end=maxlen * batch, step=maxlen).unsqueeze(-1).expand(-1, maxlen).reshape((batch * maxlen,))
    indexer = torch.arange(maxlen).repeat(batch, 1).reshape((batch*maxlen,))
    assert valid.shape == indexer.shape
        
    valid = valid + offset

    # get the sum of edges of each target tree in the batch
    sums = S[valid, indexer].reshape(batch, maxlen)

    # compute the negative log likelihood of each tree

    P = sums.sum(dim=-1)

    return P

def GE_loss(scores, features, prior):
    """Computes the Generalised Expectation Criteria equation using a given prior distribution."""
    X = torch.einsum('bij,bijf->bf', scores, features)
    Y = X - prior
    return 0.5 * (Y.unsqueeze(-2) @ Y.unsqueeze(-1)).squeeze(-1).squeeze(-1)

def entropy_loss(marginals, scores):
    """Computes an un-normalised version of the Shannon Entropy."""
    mask = scores == inf
    masked = scores.masked_fill(mask, 0.0)
    X = torch.einsum('bij,bij->b', marginals, masked).sum()
    return X
