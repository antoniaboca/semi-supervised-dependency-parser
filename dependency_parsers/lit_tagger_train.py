from lit_data_module import DataModule
from data.params import OBJECT_FILE, BATCH_SIZE
from data.params import EMBEDDING_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT, NUM_EPOCH
from pos_tagger.lit_lstm import LitLSTMTagger

import pytorch_lightning as pl

module = DataModule(OBJECT_FILE, BATCH_SIZE)

TAGSET = module.TAGSET_SIZE
model = LitLSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT, TAGSET)

trainer = pl.Trainer(max_epochs=NUM_EPOCH)
trainer.fit(model, module.train_dataloader, module.dev_dataloader)

loss_fn = model.loss_fn

print('TESTING...')
results = trainer.test(model, module.dev_dataloader, verbose=True)
print(results)
