
from torch.nn.modules import loss
from data.params import OBJECT_FILE, BATCH_SIZE, TRAIN_SPLIT, VAL_SPLIT, PARAM_FILE
from data.params import EMBEDDING_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT, NUM_EPOCH
from lit_lstm import LitLSTMTagger, DataModule

import pytorch_lightning as pl

module = DataModule(OBJECT_FILE, BATCH_SIZE, PARAM_FILE)

TAGSET = module.TAGSET_SIZE
model = LitLSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT, TAGSET)

trainer = pl.Trainer(max_epochs=NUM_EPOCH)
trainer.fit(model, module.train_dataloader, module.dev_dataloader)

loss_fn = model.loss_fn

#import matplotlib.pyplot as plt

#plt.plot(loss_fn)
#plt.show()

print('TESTING...')
results = trainer.test(model, module.dev_dataloader, verbose=True)
print(results)
