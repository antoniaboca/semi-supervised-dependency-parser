

# Evaluation of a semi-supervised expectation-based dependency parser

This project represents the code base for the dissertation of Antonia-Irina Boca, developed for the completion of the Computer Science Tripos Part II at the University of Cambridge.

The project implements a dependency parser following the architecture of Dozat & Manning (2017) that can be trained using three techniques:
1. A supervised technique using cross entropy as a loss function
2. A semi-supervised technique using Shannon Entropy as a loss function
3. A semi-supervised technique using the Generalised Expectation Criteria (Druck et al. 2009; Mann and McCallum, 2010) as a loss function

## Installation
Make sure to run the code within a `conda` environment for **python 3.7**. To create a new conda environment, run:
```
conda create -n parser python=3.7
```
Then, to activate the environment, run:
```
conda activate parser
```
To install all required dependencies, use the following comamnds, specifing the necessary libraries and their versions:
```
python3 -m pip install -r requirements.txt
python3 -m install -qU git+https://github.com/harvardnlp/pytorch-struct
python3 -m pip install transformers==4.10
```

## Data loading
The project accepts `.conllu` files that it turns into train, validation and test datasets. To read a `.conllu` file and create an embeddings vector, run the `load` command:
```
python3 ./main.py load --train-data <path/to/train/file> --validation-data <path/to/val/file> --testing-data <path/to/test/file> --embeddings <path/to/embeddings/file>
```
This will create a `cache.pickle` file that is used by the parser for training. 

###Low-resource simulation

To create datasets for semi-supervised training, run:
```
python3 ./main.py unlabelled_bucket --train-data <path/to/train/file> --validation-data <path/to/val/file> --testing-data <path/to/test/file> --embeddings <path/to/embeddings/file>
```
And:
```
python3 ./main.py labelled_bucket --train-data <path/to/train/file> --validation-data <path/to/val/file> --testing-data <path/to/test/file> --embeddings <path/to/embeddings/file>
```

This will create `.pickle` files containing random samples of different sizes from the original dataset. 

## Training

The entry point for the training of the parser is `main.py`, that accepts command line arguments to set the hyperparameters of the neural network model. 

To check all possible command line settings, run:
```
python3 ./main.py --help
```
Alternatively, you can check [this section](#full-list-of-command-line-arguments).

### Supervised training
To run the model with default parameters in the supervised context:
```
python3 ./main.py train 
```

### Semi-supervised training with GE criteria
To run the model with default parameters in the semi-supervised context, choose one of the `.pickle` files created for the low-resource simulation, and then run:
```
python3 ./main.py train --semi --model [ge, entropy] --file <chosen/pickle/file> 
```

## Analysis
To get statistics on the distribution of Parts-of-Speech tags in the original `.conllu` file, run:
```
python3 ./main.py analysis --file <path/to/test/data> --all-coverage --get-top-edges
```

## Full list of command line arguments
When running `python3 ./main.py`, we have the following command line arguments (option parameters have default values):
```
positional arguments:
{
  load                 read and process conllu files
  train                train a model
  loop                 train 5 models on 5 dataset sizes
  labelled_bucket      create low-resource files for the supervised context
  unlabelled_bucket    create low-resource files for the semi-supervised
                       context
  analysis             run analysis on dataset of pos tag edges
}
  
optional arguments:
  -h, --help            show this help message and exit

Model parameters:
  --batch-size BATCH_SIZE
                        batch size of the training process
  --hidden-dim HIDDEN_DIM
                        hidden dimension of the LSTM
  --num-layers NUM_LAYERS
                        number of layers in the LSTM
  --arc-dim ARC_DIM     arc dimension of the biaffine layer
  --lab-dim LAB_DIM     label dimension of the biaffine layer
  --lstm-dropout LSTM_DROPOUT
                        Dropout rate inside the LSTM
  --epochs EPOCHS       Number of epochs to train on
  --lr LR               Learning rate of the optimizer
  --linear-dropout LINEAR_DROPOUT
                        Add dropout to the linear layers
  --file FILE           File containing formatted input data
  --cle                 Use the chuliu-edmonds algorithm to create trees for
                        testing phase
  --semi                Train in the semi-supervised context
  --labelled-size LABELLED_SIZE
                        Number of labelled sentences in the semi-supervised
                        context
  --semi-labelled-batch SEMI_LABELLED_BATCH
                        Number of labelled sentences in a batch for the semi
                        supervised context
  --ge-only             Use only unlabelled data to train the GE parser
  --oracle              Use oracle distribution for GE criteria context
  --name NAME
                        name of model for pytorch logs
  --labelled-loss-ratio LABELLED_LOSS_RATIO
                        ratio to be used when computing split loss
  --tag-type {xpos,upos}
  --model {ge,entropy}

Dataset size:
  --train TRAIN         Max amount of sentences to load for training
  --validation VALIDATION
                        Maximum sentences to load for validation
  --test TEST           Maximum sentences to test on
  --embedding-dim EMBEDDING_DIM
                        The dimension of the pretrained embeddings
  --save-to-pickle-file SAVE_TO_PICKLE_FILE
                        File where to save formatted input data

Data loading arguments:
  --train-data TRAIN_DATA
                        Conllu file to load training sentences from
  --validation-data VALIDATION_DATA
                        Conllu file to load validation sentences from
  --testing-data TESTING_DATA
                        Conllu file to load testing sentences from
  --embeddings EMBEDDINGS
                        Text file to load embeddings from
  --limit-sentence-size LIMIT_SENTENCE_SIZE
                        Filter sentences based on how many tokens they have.

Analysis arguments:
  --all-coverage        Determine all levels of coverage
  --set-coverage SET_COVERAGE
                        Return number of tags necessary for given coverage
  --get-top-edges GET_TOP_EDGES
                        Get the top number of POS tag edges
  --get-coverage-all-edges
  --min-occurence MIN_OCCURENCE
                        Minimum frequency of an edge to be taken into account
  --sentence-length SENTENCE_LENGTH
                        Maximum sentence length for analysis
  --step STEP
                        next coverage to check = prev coverage + step
```