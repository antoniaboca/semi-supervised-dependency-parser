import torch
import pytorch_lightning as pl

from lit_model import BiAffineParser
from lit_module import LitModel
from lit_data_module import DataModule

OBJECT_FILE = 'dependency_parsers/data/cache.pickle'
PARAM_FILE = 'dependency_parsers/data/parameters.pickle'
BATCH_SIZE = 32

module = DataModule(OBJECT_FILE, BATCH_SIZE, PARAM_FILE)
embeddings = module.embeddings
model = BiAffineParser(embeddings, len(embeddings), 128, 128, 128, 0.1, 20)
pl_model = LitModel(model)
trainer = pl.Trainer(max_epochs=5)
trainer.fit(pl_model, module.train_dataloader)

count = 0
for batch in module.train_dataloader:
    count +=1 
    if count <= 100:
        continue
    
    if count > 101:
        break
    parent_scores = model(batch)
    targets = batch['parents']
    parents = torch.argmax(parent_scores, dim=-2)
    for idx in range(len(batch)):
        print('For idx {}:\n Predict: {}\n Targets: {}\n'.format(idx, parents, targets))

