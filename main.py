#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import config

config.add_option('-m', '--mode', dest='mode', default='train', type='string', help='[train|eval|pred]', action='store')
config.add_option('--seed', dest='seed', default=1, type='int', help='torch random seed', action='store')

def train():
    pass

if __name__ == '__main__':
    config.parse_args()
    torch.manual_seed(config.get_option('seed'))
    mode = config.get_option('mode')
    if mode == 'train':
        train()
    else:
        NotImplementedError()
