#!/usr/bin/env python
# -*- coding: utf-8 -*-

class FScore(object):
    def __init__(self):
        self.t, self.gold, self.pred = 0., 0., 0.

    def update(self, predict, gold):
        assert predict.size() == gold.size()
        assert predict.dim() == 2
        n, m = predict.size()
        for i in range(n):
            for j in range(m):
                self.t += 1 if predict.data[i][j] == gold[i][j] and gold[i][j] != 0 else 0
                self.gold += 1 if gold[i][j] != 0 else 0
                self.pred += 1 if predict.data[i][j] != 0 else 0


    @property
    def p(self):
        return 0 if self.pred == 0. else self.t / self.pred

    @property
    def r(self):
        return 0 if self.gold == 0. else self.t / self.gold

    @property
    def f(self):
        return 0 if self.p + self.r == 0. else 2. * (self.p * self.r) / (self.p + self.r)

    def __lt__(self, that):
        return self.f < that.f

    def __gt__(self, that):
        return self.f > that.f

    def __str__(self):
        return 'P:{:.2f}% R:{:.2f}% F:{:.2f}%'.format(self.p * 100., self.r * 100., self.f * 100.)