from torch.nn.modules.loss import CrossEntropyLoss
from .lit_data_module import DataModule
from .semi_supervised_parser.semi_supervised_lstm import LitSemiSupervisedLSTM
from .semi_supervised_parser.transfer_learning import LitSemiTransferLSTM

import torch
import pytorch_lightning as pl

def semisupervised_train(args):

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

    module.prepare_data()
    module.setup(stage='fit')

    TAGSET = module.TAGSET_SIZE
    LABSET = module.LABSET_SIZE
    embeddings = module.embeddings
    prior = module.get_prior()

    transfer = LitSemiTransferLSTM(args, prior)

    model = LitSemiSupervisedLSTM(embeddings, prior, EMBEDDING_DIM, HIDDEN_DIM, NUM_LAYERS, LSTM_DROPOUT, LINEAR_DROPOUT,
                    ARC_DIM, LAB_DIM, LABSET, LR, 'cross', args.cle, args.ge_only)

    early_stop = pl.callbacks.EarlyStopping(monitor='validation_loss', min_delta=0.01, patience=5, mode='min')
    
    logger = pl.loggers.TensorBoardLogger('semisupervised_logs/')
    if args.model_name is not None:
        logger = pl.loggers.TensorBoardLogger('semisupervised_size_logs/', sub_dir=args.model_name)
        
    trainer = pl.Trainer(max_epochs=NUM_EPOCH, logger=logger, log_every_n_steps=10, flush_logs_every_n_steps=50, callbacks=[early_stop])
    trainer.fit(model, module)

    print('TESTING...')
    results = trainer.test(model, module, verbose=True)
    print(results)

def size_loop(args):
    sizes = [100, 500, 1000, 4000]
    for size in sizes:
        args.train = size
        args.file = 'unlabelled' + str(size) + '.pickle'
        args.model_name = 'semisupervised_chuliu_edmonds_' + str(size)
        semisupervised_train(args)
