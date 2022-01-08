from torch.nn.modules.loss import CrossEntropyLoss
from .lit_data_module import DataModule
from .biaffine_parser.biaffine_lstm import LitLSTM

import torch
import pytorch_lightning as pl

def edge_count(set):
    tree_edges = {}
    graph_edges = {}

    for sentence in set:
        for idx1 in range(len(sentence[0])):
            for idx2 in range(len(sentence[0])):
                if idx1 == idx2:
                    continue

                tag1 = sentence[1][idx1]
                tag2, parent2 = sentence[1][idx2], sentence[2][idx2]
                
                if (tag1, tag2) not in graph_edges:
                    graph_edges[(tag1, tag2)] = 0
                graph_edges[(tag1, tag2)] += 1

                if parent2 == idx1:
                    if (tag1, tag2) not in tree_edges:
                        tree_edges[(tag1, tag2)] = 0
                    tree_edges[(tag1, tag2)] += 1

    return tree_edges, graph_edges

def top20(tree, graph):
    distribution = {}
    for edge in graph.keys():
        if graph[edge] < 10:
            continue
        distribution[edge] = tree.get(edge, 0) / graph[edge]
    top = []
    for key, value in distribution.items():
        top.append((value, key))
    top.sort(reverse=True)
    distribution = {}
    for value, key in top[:20]:
        distribution[key] = value
    return distribution

def biaffine_train(args):

    BATCH_SIZE = args.batch_size
    EMBEDDING_DIM = args.embedding_dim
    HIDDEN_DIM = args.hidden_dim
    NUM_LAYERS = args.num_layers
    ARC_DIM = args.arc_dim
    LAB_DIM = args.lab_dim
    LSTM_DROPOUT = args.lstm_dropout
    LINEAR_DROPOUT = args.linear_dropout
    NUM_EPOCH = args.epochs
    DATA_FILE = args.file
    LR = args.lr
    TRAIN_SIZE = args.train
    VAL_SIZE = args.validation
    TEST_SIZE = args.test

    module = DataModule(DATA_FILE, BATCH_SIZE, EMBEDDING_DIM, 
                        TRAIN_SIZE, VAL_SIZE, TEST_SIZE, args)

    TAGSET = module.TAGSET_SIZE
    LABSET = module.LABSET_SIZE
    embeddings = module.embeddings
    if args.semi:
        labelled = module.labelled
        print('Creating prior distribution for the semi-supervised context...')
        tree_edges, graph_edges = edge_count(labelled)
        features20 = top20(tree_edges, graph_edges)

    model = LitLSTM(embeddings, EMBEDDING_DIM, HIDDEN_DIM, NUM_LAYERS, LSTM_DROPOUT, LINEAR_DROPOUT,
                    ARC_DIM, LAB_DIM, LABSET, LR, 'cross', args.cle)

    early_stop = pl.callbacks.EarlyStopping(monitor='validation_loss', min_delta=0.01, patience=5, mode='min')
    
    logger = pl.loggers.TensorBoardLogger('my_logs/')
    if args.model_name is not None:
        logger = pl.loggers.TensorBoardLogger('train_size_logs/', sub_dir=args.model_name)
        
    trainer = pl.Trainer(max_epochs=NUM_EPOCH, logger=logger, log_every_n_steps=10, flush_logs_every_n_steps=50, callbacks=[early_stop])
    trainer.fit(model, module.train_dataloader, module.dev_dataloader)

    #import matplotlib.pyplot as plt
    #plt.plot(model.log_loss)
    #plt.show()

    print('TESTING...')
    results = trainer.test(model, module.test_dataloader, verbose=True)
    print(results)

def size_loop(args):
    sizes = [1000, 2000, 4000, 8000, 12000]
    for size in sizes:
        args.train = size
        args.file = 'train' + str(size) + '.pickle'
        args.model_name = 'train_chuliu_edmonds_' + str(size)
        biaffine_train(args)
