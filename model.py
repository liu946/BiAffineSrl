#!/usr/bin/env python
# -*- coding: utf-8 -*-

import config
config.add_option('--dropout_rate', dest='dropout_rate', default=0., type='float', help='model dropout', action='store')
config.add_option('--activate', dest='activate', default='tanh', type='string', help='activate functions', action='store')

__all__ = ['Model']

class BaseModel(object):
    def __init__(self, *args, **kwargs):
        self.dropout = config.get_option('dropout')
        self.activate = config.get_option('activate')

    def biaffine(self):
        pass

    def train(self, sentence):
        NotImplementedError()

    def predict(self, sentence):
        NotImplementedError()

    def fit(self, trainset):
        NotImplementedError()

class Model(BaseModel):
    def __init__(self, *args, **kwargs):
        self.word_vocab = kwargs.pop('word_vocab', None)
        self.char_vocab = kwargs.pop('char_vocab', None)
        self.pos_vocab = kwargs.pop('pos_vocab', None)
        self.rel_vocab = kwargs.pop('rel_vocab', None)
        self.srl_rel_vocab = kwargs.pop('srl_rel_vocab', None)
        super().__init__(self, *args, **kwargs)
    