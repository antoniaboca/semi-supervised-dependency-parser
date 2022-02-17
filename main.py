import argparse
from dependency_parsers.lit_biaffine_train import biaffine_train, size_loop
from dependency_parsers.lit_semi_supervised_train import semisupervised_train
from dependency_parsers.data.load import file_save, bucket_loop, bucket_unlabelled_loop

def main():
    parser = argparse.ArgumentParser(description = "Supervised biaffine dependency parser")

    parser.add_argument('mode', type=str, choices=['load', 'train', 'loop', 'labelled_bucket', 'unlabelled_bucket'])

    model = parser.add_argument_group('Model parameters')
    model.add_argument('--batch-size', type=int, default=32)
    model.add_argument('--hidden-dim', type=int, default=128, help='hidden dimension of the LSTM')
    model.add_argument('--num-layers', type=int, default=2, help='number of layers in the LSTM')
    model.add_argument('--arc-dim', type=int, default=128, help='arc dimension of the biaffine layer')
    model.add_argument('--lab-dim', type=int, default=128, help='label dimension of the biaffine layer')
    model.add_argument('--lstm-dropout', type=float, default=0.1)
    model.add_argument('--epochs', type=int, default=50, help='Number of epochs to train on')
    model.add_argument('--lr', type=float, default=1e-3, help='Learning rate of the optimizer')
    model.add_argument('--linear-dropout', type=float, default=0.1, help='Add dropout to the linear layers')
    model.add_argument('--file', type=str, default='dependency_parsers/data/cache.pickle', help='File containing formatted input data')
    model.add_argument('--cle', action='store_true', help='Use the chuliu-edmonds algorithm to create trees for testing phase')
    model.add_argument('--semi', action='store_true', help='Use the semi-supervised dependency parser')
    model.add_argument('--transfer', action='store_true', help='Use the weights of a supervised parser')
    model.add_argument('--labelled-size', type=int, default=20000, help='Number of labelled sentences in the semi-supervised context')
    model.add_argument('--semi-labelled-batch', type=int, default=8, help='Number of labelled sentences in a batch for the semi supervised context')
    model.add_argument('--ge-only', action='store_true', help='Use only unlabelled data to train the semi-supervised parser')
    model.add_argument('--oracle', action='store_true', help='Use oracle prior distribution for semi-supervised context')
    model.add_argument('--name', type=str, default=None)
    model.add_argument('--labelled-loss-ratio', type=float, default=None)
    
    data = parser.add_argument_group('Dataset size')
    data.add_argument('--train', type=int, default=20000, help='Max amount of sentences to load for training')
    data.add_argument('--validation', type=int, default=4000, help='Maximum sentences to load for validation')
    data.add_argument('--test', type=int, default=4000, help='Maximum sentences to test on')
    data.add_argument('--embedding-dim', type=int, default=100, help='The dimension of the pretrained embeddings')
    data.add_argument('--save-to-pickle-file', type=str, default='dependency_parsers/data/cache.pickle', help='File where to save formatted input data')

    loader = parser.add_argument_group('Data loading arguments')
    loader.add_argument('--train-data', type=str, default='./en_ewt-ud-train.conllu')
    loader.add_argument('--validation-data', type=str, default='./en_ewt-ud-dev.conllu')
    loader.add_argument('--testing-data', type=str, default='./en_ewt-ud-test.conllu')
    loader.add_argument('--embeddings', type=str, default='./glove.6B.100d.txt')
    loader.add_argument('--limit-sentence-size', type=int, default=0, help='Filter sentences based on how many tokens they have.')

    args = parser.parse_args()
    args.model_name = None

    if args.mode == 'labelled_bucket':
        bucket_loop(args)
    if args.mode == 'unlabelled_bucket':
        bucket_unlabelled_loop(args)

    if args.mode == 'load':
        file_save(args)
    if args.mode == 'train':
        if args.semi is False:
            biaffine_train(args)
        else:
            semisupervised_train(args)
            
    if args.mode == 'loop':
        size_loop(args)

if __name__ == '__main__':
    main()