#!/usr/bin/env python
# -*- coding: utf-8 -*-
import codecs
import os
import numpy
from collections import Counter
import config
import math

config.add_option('--recount', dest='recount', default=False, help='If there need to be recount the vocab', action='store_true')

class BaseVocab(object):
    _special_tokens = []
    min_occur_count = 0
    max_rank = math.inf
    def __init__(self, *args, **kwargs):
        self._name = kwargs.pop('name', self.__class__.__name__)
        self._filename = os.path.join(config.get_option('save'), self._name + '.txt')
        self._complete = False
        self._str2idx = zip(self._special_tokens, range(len(self._special_tokens)))
        self._idx2str = zip(range(len(self._special_tokens)), self._special_tokens)
        self._tok2idx = self._str2idx
        self._counts = None
        recount = config.get_option('recount')
        if not recount and os.path.isfile(self.filename):  # read file from existed vocab file.
            self.load()

    def convert(self, token):
        return self._tok2idx[token]

    def count(self, sentence):
        if self._complete:
            RuntimeError('Completed vocab can not read more data.')

    def add_one(self, token):
        self.counts[token] += 1

    def index(self, sentence):
        if not self._complete:
            RuntimeError('Uncompleted vocab can not be indexed.')

    def complete(self):
        for token, count in self.sorted_counts(self.counts):
            if ((count >= self.min_occur_count) and
                        token not in self and
                    (not self.max_rank or len(self) < self.max_rank)):
                self[token] = len(self)
        self._complete = True
        return

    def is_complete(self):
        return self._complete

    # def merge(self, vocab):
    #     raise NotImplementedError()

    def load(self):
        if self._complete:
            RuntimeError('Completed vocab can not read more data.')
        with codecs.open(self.filename, encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                try:
                    line = line.strip()
                    if line:
                        line = line.split('\t')
                        token, count = line
                        self.counts[token] = int(count)
                except:
                    raise ValueError('File %s is misformatted at line %d' % (self.name, line_num + 1))
        self.complete()
        return

    def dump(self):
        with codecs.open(self.filename, 'w', encoding='utf-8') as f:
            for word, count in self.sorted_counts(self.counts):
                f.write('%s\t%d\n' % (word, count))
        return

    @staticmethod
    def sorted_counts(counts):
        return sorted(counts.most_common(), key=lambda x: (-x[1], x[0]))

    @property
    def filename(self):
        return self._filename
    @property
    def name(self):
        return self._name
    @property
    def counts(self):
        return self._counts
    @property
    def conll_idx(self):
        return self._conll_idx

    def __setitem__(self, key, value):
        if isinstance(key, str):
            self._str2idx[key] = value
            self._idx2str[value] = key
        elif isinstance(key, int):
            self._idx2str[key] = value
            self._str2idx[value] = key
        elif hasattr(key, '__iter__') and hasattr(value, '__iter__'):
            for k, v in zip(key, value):
                self[k] = v
        else:
            raise ValueError('keys and values to BaseVocab.__setitem__ must be (iterable of) string or integer')

    def __contains__(self, key):
        if isinstance(key, str):
            return key in self._str2idx
        elif isinstance(key, int):
            return key in self._idx2str
        else:
            raise ValueError('key to BaseVocab.__contains__ must be string or integer')
        return

    def __setattr__(self, name, value):
        if name in ('_str2idx', '_idx2str', '_str2idxs'):
            value = dict(value)
        elif name == '_counts':
            value = Counter(value)
        super(BaseVocab, self).__setattr__(name, value)
        return

    def __len__(self):
        return len(self._str2idx)

    def __iter__(self):
        return (key for key in sorted(self._str2idx, key=self._str2idx.get))

class WordLevelVocab(BaseVocab):
    def count(self, sentence):
        super().count(sentence)
        for line_num, line in enumerate(sentence):
            token = line[self.conll_idx]
            self.add_one(token)

    def index(self, sentence):
        super().index(sentence)
        return numpy.array([self.convert(line[self._conll_idx]) for line in sentence])

config.add_option('--word_cased', dest='word_cased', default=False, help='If case sensitive', action='store_true')

class UnkableVocab(WordLevelVocab):
    _special_tokens = ['<unk>']
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.UNK = self._str2idx['<unk>']

    def convert(self, token):
        return self._tok2idx.get(token, self.UNK)

class CaseAbleVocab(UnkableVocab):
    def __init__(self, *args, **kwargs):
        self.case = config.get_option('word_cased')
        super().__init__(*args, **kwargs)

    def convert(self, token):
        token = token if self.case else token.lower()
        return super().convert(token)

    def add_one(self, token):
        token = token if self.case else token.lower()
        return super().add_one(token)

    def __setitem__(self, key, value):
        if not self.case:
            if isinstance(key, str):
                key = key.lower()
            if isinstance(value, str):
                value = value.lower()
        super().__setitem__(key, value)

    def __contains__(self, key):
        if not self.case and isinstance(key, str):
            key = key.lower()
        return super().__contains__(key)

class WordVocab(CaseAbleVocab):
    min_occur_count = 2
    _conll_idx = 1
class LemmaVocab(UnkableVocab):
    _conll_idx = 2
class TagVocab(UnkableVocab):
    _conll_idx = 4

class ConstVocab(BaseVocab):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.complete()

    def count(self, sentence):
        pass

    def index(self, sentence):
        super().index(sentence)


class PredictVocab(ConstVocab):
    _conll_idx = 12
    _special_tokens = ['_', 'Y']

    def index(self, sentence):
        super().index(sentence)
        return [i for i, line in enumerate(sentence) if line[self._conll_idx] == 'Y']

class SentenceLevelVocab(BaseVocab):

    def count(self, sentence):
        super().count(sentence)
        for line_num, line in enumerate(sentence):
            tokens = line[self.conll_idx]
            for token in tokens:
                self.add_one(token)

    def index(self, sentence):
        super().index(sentence)
        return numpy.array([[self.convert(token) for token in tokens[self._conll_idx]] for tokens in sentence])

class SemTagVocab(SentenceLevelVocab):
    _conll_idx = slice(14, None)
