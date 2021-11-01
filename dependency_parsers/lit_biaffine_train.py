from lit_data_module import DataModule
from data.params import OBJECT_FILE, BATCH_SIZE, PARAM_FILE
from data.params import EMBEDDING_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT, NUM_EPOCH, ARC_DIM
from biaffine_parser.biaffine_lstm import LitLSTM

import pytorch_lightning as pl

module = DataModule(OBJECT_FILE, BATCH_SIZE, PARAM_FILE)

TAGSET = module.TAGSET_SIZE
model = LitLSTM(EMBEDDING_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT, ARC_DIM)

trainer = pl.Trainer(max_epochs=NUM_EPOCH)
trainer.fit(model, module.train_dataloader, module.dev_dataloader)

loss_fn = model.loss_fn

#import matplotlib.pyplot as plt

#plt.plot(loss_fn)
#plt.show()

print('TESTING...')
results = trainer.test(model, module.dev_dataloader, verbose=True)
print(results)