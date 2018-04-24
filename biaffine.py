#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import math
from torch import nn, autograd
from torch.nn import Parameter

class BiAffineModel(nn.Module):
    r"""Applies a bilinear transformation to the incoming data:
    :math:`y = x_1 * A * x_2 + b`

    Args:
        in1_features: size of each first input sample
        in2_features: size of each second input sample
        out_features: size of each output sample
        bias: If set to False, the layer will not learn an additive bias.
            Default: ``True``

    Shape:
        - Input: :math:`(N1, in1\_features)`, :math:`(N2, in2\_features)`
        - Output: :math:`(N1, N2, out\_features)`

    Attributes:
        weight: the learnable weights of the module of shape
            (out_features x in1_features x in2_features)
        bias:   the learnable bias of the module of shape (out_features)

    Examples::

        >>> m = BiAffineModel(20, 30, 40)
        >>> input1 = autograd.Variable(torch.randn(45, 20))
        >>> input2 = autograd.Variable(torch.randn(55, 30))
        >>> output = m(input1, input2)
        >>> print(output.size())
        torch.Size([45, 55, 40])
    """

    def __init__(self, in1_features, in2_features, out_features, bias=True):
        super().__init__()
        self.in1_features = in1_features
        self.in2_features = in2_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in1_features, in2_features))

        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input1, input2):
        output = []

        # compute output scores:
        for k, w in enumerate(self.weight):
            buff = torch.mm(input1, w)
            rel_score_one_role = buff.mm(input2.transpose(0, 1))
            output.append(rel_score_one_role.view(rel_score_one_role.size() + (1, )))

        output = torch.cat(output, -1)

        if self.bias is not None:
            output.add_(self.bias.expand_as(output))

        return output.transpose(1, 2)

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'in1_features=' + str(self.in1_features) \
            + ', in2_features=' + str(self.in2_features) \
            + ', out_features=' + str(self.out_features) \
            + ', bias=' + str(self.bias is not None) + ')'


if __name__ == '__main__':
    m = BiAffineModel(20, 30, 40)
    input1 = autograd.Variable(torch.randn(45, 20))
    input2 = autograd.Variable(torch.randn(55, 30))
    output = m(input1, input2)
    print(output.size())

