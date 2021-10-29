import pickle
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.nn.modules import loss
import torch.optim as optim 

from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_packed_sequence, pad_sequence

from lstm import LSTMTagger
from data.processor import collate_fn_padder
from data.params import OBJECT_FILE, BATCH_SIZE, EMBEDDING_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT, NUM_EPOCH, PARAM_FILE

TAGSET_SIZE = 0
print('Load tagset param...')
with open(PARAM_FILE, 'rb') as file:
    dict = pickle.load(file)
    TAGSET_SIZE = dict['TAGSET_SIZE']
print('Loaded.')

print('Load curated set from cache...')
with open(OBJECT_FILE, 'rb') as file:
    embedded_set = pickle.load(file)
print('Loaded curated set: {} tuples.'.format(len(embedded_set)))

print('Initialize a data loader...')
train_dataloader = DataLoader(embedded_set, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn_padder)


print('Initialize the model...')
model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT, TAGSET_SIZE)
loss_function = nn.CrossEntropyLoss(ignore_index=0)
optimizer = optim.Adam(model.parameters(), lr=0.1) 

print('Start training the model...')

loss_fn = []
prev_loss = 0.0

delta_loss = []

for epoch in range(NUM_EPOCH):
    loss_value = 0
    instances = 0
    num_correct = 0
    total = 0
    for train_batch in tqdm(train_dataloader):
        targets, _ = pad_packed_sequence(train_batch['tags'], batch_first=True)
        #print('Model batch {};'.format(instances))

        model.zero_grad()
        tag_scores = model(train_batch)
        batch_size, sent_len, num_tags = tag_scores.shape
        total_loss = loss_function(
            tag_scores.reshape(batch_size * sent_len, num_tags),
            targets.reshape(batch_size * sent_len)
        )
        tags = tag_scores.argmax(-1)
        num_correct += torch.count_nonzero((tags == targets) * (targets != 0))
        total += torch.count_nonzero(targets)

        loss_value += total_loss.item()
        loss_fn.append(total_loss.item())
        instances += 1

        total_loss.backward()
        optimizer.step()
    
    print('Average loss value for epoch {}: {}'.format(epoch, loss_value/instances))
    print('Training accuracy for epoch {}:  {}'.format(epoch, num_correct/total))

    print('Difference between previous loss and this loss: {}'.format(prev_loss - loss_value))
    delta_loss.append(prev_loss - loss_value)
    prev_loss = loss_value

with torch.no_grad():
    print('AFTER TRAINING')

    correct = 0
    total = 0
    
    gen = iter(train_dataloader)
    while True:    
        try:
            train_batch = next(gen)
            tag_scores = model(train_batch)

            sentences, _ = pad_packed_sequence(train_batch['sentence'], batch_first=True)
            embeddings, _ = pad_packed_sequence(train_batch['embedding'], batch_first=True)
            targets, _ = pad_packed_sequence(train_batch['tags'], batch_first=True)

            for sentence_idx in range(len(sentences)):
                sentence = sentences[sentence_idx]
                
                for idx in range(len(sentence)):
                    if sentence[idx] < 3:
                        continue

                    max_idx = 0
                    for tag_idx in range(len(tag_scores[sentence_idx][idx])):
                        if tag_scores[sentence_idx][idx][tag_idx] > tag_scores[sentence_idx][idx][max_idx]:
                            max_idx = tag_idx

                    total += 1
                    if max_idx == targets[sentence_idx][idx].item():
                        correct += 1
                    
        except StopIteration:
            break

import matplotlib.pyplot as plt

print(delta_loss)
plt.plot(delta_loss)
plt.show()