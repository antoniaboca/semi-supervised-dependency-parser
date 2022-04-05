from torch.nn.modules.loss import CrossEntropyLoss

from dependency_parsers.parsers.entropy_parser import LitEntropyLSTM
from .lit_data_module import DataModule
from .parsers.ge_criteria_parser import LitSemiSupervisedLSTM

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
    
    if args.labelled_loss_ratio is None:
        args.labelled_ratio = args.semi_labelled_batch / (args.semi_labelled_batch + args.batch_size)
    else:
        args.labelled_ratio = args.labelled_loss_ratio

    module = DataModule(DATA_FILE, BATCH_SIZE, EMBEDDING_DIM, 
                        TRAIN_SIZE, VAL_SIZE, TEST_SIZE, args)

    module.prepare_data()
    module.setup(stage='fit')

    TAGSET = module.TAGSET_SIZE
    LABSET = module.LABSET_SIZE
    args.num_labels = LABSET
    
    embeddings = module.embeddings
    prior = module.get_prior()
    order20 = module.order20
    vocab = module.vocabulary


    if args.model == 'ge':
        model = LitSemiSupervisedLSTM(embeddings, prior, vocab, order20, args)
    else:
        model = LitEntropyLSTM(embeddings, args)

    early_stop = pl.callbacks.EarlyStopping(monitor='validation_loss', min_delta=0.01, patience=10, mode='min')
    
    logger = pl.loggers.TensorBoardLogger('logs/semisupervised_logs/')
    if args.name is not None:
        logger = pl.loggers.TensorBoardLogger('logs/evaluation_logs/', sub_dir=args.name)
        
    trainer = pl.Trainer(max_epochs=NUM_EPOCH, logger=logger, log_every_n_steps=1, flush_logs_every_n_steps=1, callbacks=[early_stop])
    trainer.fit(model, module)

    print('TESTING...')
    results = trainer.test(model, module, verbose=True)
    print(results)

def size_loop_semi_supervised(args):
    sizes = [1000, 4000]
    for size in sizes:
        args.entropy = False
        args.semi = True
        args.limit_sentence_size = 0
        args.transfer = False
        args.ge_only = False
        args.train = size
        args.file = 'unlabelled' + str(size) + '.pickle'
        args.model_name = 'entropy-no-length-limit-' + str(size)
        semisupervised_train(args)

        
        args.limit_sentence_size = 12
        args.file = 'limit-12-' + str(size) + '.pickle'
        args.model_name = 'entropy-limit-12-' + str(size)
        semisupervised_train(args)