import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from torch.nn import Linear, LSTM

class Biaffine(nn.Module):
    """Module that performs a biaffine transformation on the scores"""

    def __init__(self, arc_dim, output_dim, from_pretrained=None):
        """Constructor method for the Biaffine module.

        Args:
            arc_dim (int): Dimension of the matrix transformation
            output_dim (int): Dimension of the output value
            from_pretrained (numpy.ndarray, optional): If present, the biaffine 
                matrix will set its values to the ones in 'from_pretrained'. Defaults to None.
        """

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
        """Defines the computation performed at every call of the model.

        Args:
            head (torch.Tensor): The HEAD representation 
            dep (torch.Tensor): The DEPENDANT representation

        Returns:
            float: Numerical value representing the score of the representation.
        """

        head = head.unsqueeze(1)
        dep = dep.unsqueeze(1)
        scores = head @ self.W @ dep.transpose(-1,-2)
        return scores.squeeze(1)

    def reset_parameters(self):
        """Sets the parameters of the biaffine matrix to an initial uniform random distribution."""
        nn.init.xavier_uniform_(self.W)

class MLP(nn.Module):
    """A Perceptron module composed of a Linear layer, a ReLU function and a Dropout layer.

    Layers:
        dropout (nn.Dropout): Dropout to be applied to the input
        linear_h (nn.Linear): Linear layer to extract the HEAD representation
        linear_d (nn.Linear): Linear layer to extract the DEPENDANT representation
        score_h (nn.Linear): Linear layer to extract a score from the HEAD representation
        score_d (nn.Linear): Linear layer to extract a score from the DEPENDANT representation
    """

    def __init__(self, hidden_dim, linear_dropout, out_dim):
        """Constructor method for the MLP class.

        Args:
            hidden_dim (int): dimension to be used in the first Linear layer
            linear_dropout (float): The probability of dropout 
            out_dim (int): dimension to be used in the second Linear layer
        """

        super().__init__()

        # Dropout layer
        self.dropout = nn.Dropout(linear_dropout)

        # Arc Linear Layer
        self.linear_h = Linear(hidden_dim, out_dim)
        self.linear_d = Linear(hidden_dim, out_dim)

        self.score_h = Linear(out_dim, 1)
        self.score_d = Linear(out_dim, 1)

    def forward(self, x):
        """Defines the computation performed at every call of the model.

        Args:
            x (torch.Tensor): torch.Tensor containing the input to the computation 

        Returns:
            tuple: A tuple formed of two Tensors for the HEAD and the DEPENDANT and
            two numerical scores, respectively.
        """

        h_out = self.dropout(F.relu(self.linear_h(x)))
        d_out = self.dropout(F.relu(self.linear_d(x)))

        h_score = self.score_h(h_out)
        d_score = self.score_d(d_out)
        return h_out, d_out, h_score, d_score
    
    
class BiaffineLSTM(nn.Module):
    """A class that combines all of the layers of the neural network. 

    Layers:
        word_embedding (nn.Embedding): Encoder that turns tokens into embeddings
        lstm (nn.Module): The LSTM module of the neural network
        MLP_arc (nn.Module): A Perceptron module for the arc scores
        MLP_lab (nn.Module): A Perceptron module for the label scores
        arc_biaffine (nn.Module): The biaffine attention module for the arc scores
        lab_biaffine (nn.Module): The biaffine attention module for the label scores
    """

    def __init__(self, embeddings, args):
        """Constructor method for the BiaffineParser

        Args:
            embeddings (numpy.ndarray): GloVe Embeddings indexed by the 
                index of each word in the Vocabulary
            args (object): arguments passed from the command line to configure 
                the hyperparameters of the network
        """

        super().__init__()
        self.hidden_dim = args.hidden_dim
        self.arc_dim = args.arc_dim
        self.lab_dim = args.lab_dim
        self.lr = args.lr
        self.linear_dropout = args.linear_dropout
        self.num_labels = args.num_labels

        self.lstm = LSTM(
            input_size=args.embedding_dim, 
            hidden_size=args.hidden_dim,
            num_layers=args.num_layers,
            dropout=args.lstm_dropout,
            bidirectional=True
        )

        self.word_embedding = nn.Embedding.from_pretrained(torch.tensor(embeddings), padding_idx=0)

        # Linear layers 

        self.MLP_arc = MLP(self.hidden_dim * 2, self.linear_dropout, self.arc_dim)
        self.MLP_lab = MLP(self.hidden_dim * 2, self.linear_dropout, self.lab_dim)
        
        # biaffine layers
        self.arc_biaffine = Biaffine(self.arc_dim, 1)
        self.lab_biaffine = Biaffine(self.lab_dim, self.num_labels)
    
    def forward(self, x):
        """Defines the computation performed at every call to the model.

        Args:
            x (dict): A dictionary representing the input to the network. The dictionary contains:
                sentence indexes, UPOS tag indexes, XPOS tag indexes, parent indexes, label indexes,
                the length of the sentence

        Returns:
            tuple: A tuple of two torch.Tensors. First contains the arc scores for the sentence 
                second contains the label scores
        """

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