#!/usr/bin/python
# -*- coding: utf-8 -*-

from MessageFunction import MessageFunction
from UpdateFunction import UpdateFunction
from ReadoutFunction import ReadoutFunction

import torch
import torch.nn as nn
from torch.autograd import Variable

import torch.nn.functional as F


class MPNNAttn(nn.Module):
    """
        MPNN as proposed by Gilmer et al..

        This class implements the whole Gilmer et al. model following the functions Message, Update and Readout.

        Parameters
        ----------
        in_n : int list
            Sizes for the node and edge features.
        hidden_state_size : int
            Size of the hidden states (the input will be padded with 0's to this size).
        message_size : int
            Message function output vector size.
        n_layers : int
            Number of iterations Message+Update (weight tying).
        l_target : int
            Size of the output.
        type : str (Optional)
            Classification | [Regression (default)]. If classification, LogSoftmax layer is applied to the output vector.
    """

    def __init__(self, in_n, hidden_state_size, message_size, n_layers, l_target, type='regression'):
        super(MPNNAttn, self).__init__()

        # Define message
        self.m = nn.ModuleList(
            [MessageFunction('mpnn', args={'edge_feat': in_n[1], 'in': hidden_state_size, 'out': message_size})])

        # Define Update
        self.u = nn.ModuleList([UpdateFunction('mpnn',
                                               args={'in_m': message_size,
                                                     'out': hidden_state_size})])

        # Define Readout
        self.r = ReadoutFunction('mpnn',
                                 args={'in': hidden_state_size,
                                       'target': l_target})
        # self.g = g
        # equation (1)
        self.fc = nn.Linear(hidden_state_size, hidden_state_size, bias=False)
        # equation (2)
        self.attn_fc = nn.Linear(2 * hidden_state_size, 1, bias=False)
        self.reset_parameters()
        self.type = type

        self.args = {}
        self.args['out'] = hidden_state_size
        self.n_layers = n_layers

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.fc.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_fc.weight, gain=gain)


    def forward(self, g, h_in, e):

        h = []

        # Padding to some larger dimension d
        h_t = torch.cat([h_in, Variable(
            torch.zeros(h_in.size(0), h_in.size(1), self.args['out'] - h_in.size(2)).type_as(h_in.data))], 2)

        h.append(h_t.clone())

        # Layer
        for t in range(0, self.n_layers):
            # print(h[t].size())
            h_t = h[t].view(-1, h[t].size(2))
            z = self.fc(h_t)
            # print(z.shape)
            z = z.view(h[t].size(0),h[t].size(1),h[t].size(2))
            z1 = z.repeat(1, z.size(1), 1)
            z2 = z.repeat_interleave(z.size(1), dim = 1)

            zij = torch.cat((z1, z2), dim= 2)
            # print('zij', zij.size())
            eij = F.leaky_relu(self.attn_fc(zij))
            # print('eij', eij.size())
            alpha = F.softmax(eij, dim=1)
            # print('alpha', alpha.size())
            # alpha torch.Size([10, 441, 1])
            alpha= alpha.view(h[t].size(0),h[t].size(1),h[t].size(1),-1)
            z = z[:,:,None,:].expand(h[t].size(0),h[t].size(1),h[t].size(1),-1)
            h_t = torch.sum(alpha * z, dim=2)
            # h_t = h_t.view(h[t].size())
            # h_t = self.u[0].forward(h[t], h_t)
            # print(h_t.shape)


            h_t = (torch.sum(h_in, 2, keepdim = True).expand_as(h_t) > 0).type_as(h_t) * h_t
            h.append(h_t)

        # Readout
        res = self.r.forward(h)

        if self.type == 'classification':
            res = nn.LogSoftmax()(res)
        return res
