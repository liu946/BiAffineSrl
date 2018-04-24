#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import torch
from torch import nn, autograd
import config
import time
import copy
import progressbar as pb
from dataset import TrainDataSet
from model import BiAffineSrlModel
from fscore import FScore

config.add_option('-m', '--mode', dest='mode', default='train', type='string', help='[train|eval|pred]', action='store')
config.add_option('--seed', dest='seed', default=1, type='int', help='torch random seed', action='store')

def train(num_epochs = 30):
    lossfunction = nn.CrossEntropyLoss()
    trainset = TrainDataSet()
    model = BiAffineSrlModel(vocabs=trainset.vocabs)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_f = FScore()
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1), file=sys.stderr)
        print('-' * 10, file=sys.stderr)
        for phase in ['train', 'dev']:
            model.train(phase == 'train')
            running_loss = 0.0
            running_f = FScore()

            for sentence in pb.progressbar(trainset.get_set(phase)):
                model.zero_grad()
                role_p = model(*sentence['inputs'])
                _, predict = torch.max(role_p, 1)
                loss = lossfunction(role_p, autograd.Variable(sentence['targets'][0]))
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                if epoch > 28:
                    print(predict.data)
                    print(sentence['targets'][0])
                running_loss += loss.data[0]
                running_f.update(predict, sentence['targets'][0])

            print('\n{} Loss: {:.4f} {}'.format(phase, running_loss, running_f), file=sys.stderr)

            if phase == 'dev' and running_f > best_f:
                best_f = running_f
                best_model_wts = copy.deepcopy(model.state_dict())
    print('', file=sys.stderr)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60), file=sys.stderr)
    print('Best val F: {}s'.format(best_f), file=sys.stderr)

    model.load_state_dict(best_model_wts)
    return model


if __name__ == '__main__':
    config.parse_args()
    torch.manual_seed(config.get_option('seed'))
    mode = config.get_option('mode')
    if mode == 'train':
        train()
    else:
        NotImplementedError()
