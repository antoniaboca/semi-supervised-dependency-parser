

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

**Low-resource simulation**

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
