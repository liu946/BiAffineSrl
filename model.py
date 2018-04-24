#!/usr/bin/env python
# -*- coding: utf-8 -*-

import config
import torch
from torch import nn
from torch.autograd import Variable
from biaffine import BiAffineModel
config.add_option('--dropout_rate', dest='dropout_rate', default=0., type='float', help='model dropout', action='store')
config.add_option('--activate', dest='activate', default='tanh', type='string',
                  help='activate functions (relu|glu|logsigmoid|softsign|log_softmax|sigmoid)', action='store')

__all__ = ['BiAffineSrlModel']

class BaseModel(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        vocabs = kwargs.pop('vocabs')
        self.vocab = {vocab.name:vocab for vocab in vocabs}
        self.dropout = config.get_option('dropout_rate')
        self.activate = nn.functional.__getattribute__(config.get_option('activate'))

    def biaffine(self):
        pass

config.add_option('--word_dim', dest='word_dim', default=200, type='int', help='word feature dim', action='store')
config.add_option('--pos_dim', dest='pos_dim', default=50, type='int', help='pos tag feature dim', action='store')

config.add_option('--rnn_layers', dest='rnn_layers', type='int', default=2, help='rnn layer num', action='append')
config.add_option('--rnn_hidden', dest='rnn_hidden', type='int', default=100, help='set the rnn layers output dim', action='store')
config.add_option('--pred_hidden_dim', dest='pred_hidden_dim', type='int', default=50, help='set the rnn layers output dim', action='store')
config.add_option('--role_hidden_dim', dest='role_hidden_dim', type='int', default=40, help='set the rnn layers output dim', action='store')

class BiAffineSrlModel(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(self, *args, **kwargs)
        self.word_dim = config.get_option('word_dim')
        self.pos_dim = config.get_option('pos_dim')
        self.rnn_layers = config.get_option('rnn_layers')
        self.rnn_hidden = config.get_option('rnn_hidden')
        self.pred_hidden_dim = config.get_option('pred_hidden_dim')
        self.role_hidden_dim = config.get_option('role_hidden_dim')

        self.word_lookup = nn.Embedding(len(self.vocab['WordVocab']), self.word_dim)
        self.pos_lookup = nn.Embedding(len(self.vocab['TagVocab']), self.pos_dim)
        self.rnn_input_dim = self.word_dim + self.pos_dim
        self.rnn = nn.LSTM(input_size=self.rnn_input_dim,
                           hidden_size=self.rnn_hidden // 2,
                           num_layers=self.rnn_layers,
                           batch_first=True,
                           dropout=self.dropout,
                           bidirectional=True)
        self.predicate_mlp = nn.Linear(self.rnn_hidden, self.pred_hidden_dim)
        self.role_mlp = nn.Linear(self.rnn_hidden, self.role_hidden_dim)
        self.bilinear = BiAffineModel(self.role_hidden_dim, self.pred_hidden_dim, len(self.vocab['SemTagVocab']))

    def forward(self, words, lemmas, postags, predicates):
        words, lemmas, postags, predicates = Variable(words), Variable(lemmas), Variable(postags), Variable(predicates)
        word_embs = self.word_lookup(words)
        pos_embs = self.pos_lookup(postags)
        lexical_embs = torch.cat((word_embs, pos_embs), dim=1)
        # (stn_length x rnn_hidden_dim)
        rnn_out, _ = self.rnn(lexical_embs.view((1,) + lexical_embs.size()))
        rnn_out = rnn_out.view(rnn_out.size()[1:])
        # (pred_nums x pred_hidden_dim)
        pred = self.activate(self.predicate_mlp(torch.index_select(rnn_out, 0, predicates)))
        # (stn_length x role_hidden_dim)
        role = self.activate(self.role_mlp(rnn_out))
        bilinear = self.bilinear(role, pred)
        role_p = bilinear
        return role_p

