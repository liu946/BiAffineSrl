#!/usr/bin/env python
# -*- coding: utf-8 -*-
import codecs
import config
from vocab import WordVocab, LemmaVocab, TagVocab, PredictVocab, SemTagVocab
config.add_option('-T', '--train', dest='train_file', type='string', help='train data file', action='store')
config.add_option('-D', '--dev', dest='dev_file', type='string', help='evaluation data file', action='store')


class DataSet(object):
    def __init__(self, filename, vocabs = None):
        self._filename = filename
        self._vocabs = vocabs if vocabs else [WordVocab(), LemmaVocab(), TagVocab(), PredictVocab(), SemTagVocab()]
        self._establish_vocab()
        self._data = []
        self._read_data(self._filename, self._data)

    def _establish_vocab(self):
        if any([not vocab.is_complete() for vocab in self.vocabs]):
            for conll_file in [self._filename]:
                for sentence in self.iter_sentence(conll_file):
                    for vocab in self.vocabs:
                        if not vocab.is_complete():
                            vocab.count(sentence)
        for vocab in self.vocabs:
            if not vocab.is_complete():
                vocab.complete()
                vocab.dump()

    def _read_data(self, filename, empty_data_array):
        for sentence in self.iter_sentence(filename):
            empty_data_array.append([vocab.index(sentence) for vocab in self.vocabs])

    def iter_sentence(self, filename):
        with codecs.open(filename, encoding='utf-8') as f:
            sentence = []
            for line_num, line in enumerate(f):
                try:
                    line = line.strip()
                    if (line == '' or line.startswith('#') or line.startswith('1\t')) and len(sentence):
                        yield sentence
                        sentence = []
                    if line and not line.startswith('#'):
                        line = line.split('\t')
                        self.format_check(line)
                        sentence.append(line)
                except:
                    raise ValueError('File %s is misformatted at line %d' % (filename, line_num + 1))
            else:
                if not len(sentence):
                    yield sentence
        return None

    def format_check(self, line):
        assert len(line) >= 13

    @property
    def vocabs(self):
        return self._vocabs

    def __len__(self):
        return self._data.__len__()

    def __iter__(self):
        return self._data.__iter__()

    def __index__(self, index):
        return self._data[index]

class TrainDataSet(object):
    def __init__(self):
        self.train_file = config.get_option('train_file')
        self.dev_file = config.get_option('dev_file')
        self._train_set = DataSet(self.train_file)
        self._dev_set = DataSet(self.dev_file, self._train_set.vocabs)

    @property
    def vocabs(self):
        return self.train_set.vocabs
    @property
    def train_set(self):
        return self._train_set
    @property
    def dev_set(self):
        return self._dev_set

if __name__ == '__main__':
    config.parse_args()
    dataset = TrainDataSet()
    print(len(dataset.train_set))
