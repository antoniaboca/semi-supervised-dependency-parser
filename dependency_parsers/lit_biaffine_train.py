from torch.nn.modules.loss import CrossEntropyLoss
from .lit_data_module import DataModule
from .data.params import OBJECT_FILE, PARAM_FILE
from .biaffine_parser.biaffine_lstm import LitLSTM

import torch
import pytorch_lightning as pl

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

    LR = args.lr
    TRAIN_SIZE = args.train
    VAL_SIZE = args.validation
    TEST_SIZE = args.test

    module = DataModule(OBJECT_FILE, BATCH_SIZE, PARAM_FILE, EMBEDDING_DIM, 
                        TRAIN_SIZE, VAL_SIZE, TEST_SIZE)

    TAGSET = module.TAGSET_SIZE
    LABSET = module.LABSET_SIZE
    embeddings = module.embeddings
    model = LitLSTM(embeddings, EMBEDDING_DIM, HIDDEN_DIM, NUM_LAYERS, LSTM_DROPOUT, LINEAR_DROPOUT,
                    ARC_DIM, LAB_DIM, LABSET, LR, 'cross')

    trainer = pl.Trainer(max_epochs=NUM_EPOCH)
    trainer.fit(model, module.train_dataloader, module.dev_dataloader)

    #import matplotlib.pyplot as plt
    #plt.plot(model.log_loss)
    #plt.show()

    print('TESTING...')
    results = trainer.test(model, module.test_dataloader, verbose=True)
    print(results)