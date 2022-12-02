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

    def __init__(self, in_n, hidden_state_size, message_size, n_layers, l_target, method=3,num_heads=8, type='regression'):
        super(MPNNAttn, self).__init__()

        # Define message
        if num_heads == 1:
            self.m = nn.ModuleList([MPNNMultiHAttn(in_n, hidden_state_size,hidden_state_size, method, num_heads=1, merge='mean')])
        else:
            self.m = nn.ModuleList()
            self.m.append(MPNNMultiHAttn(in_n, hidden_state_size, hidden_state_size, method, num_heads, merge='cat'))
            for _ in range(1, n_layers-1):
                self.m.append(MPNNMultiHAttn(in_n, hidden_state_size*num_heads, hidden_state_size, method, num_heads, merge='cat'))
            self.m.append(MPNNMultiHAttn(in_n, hidden_state_size*num_heads, hidden_state_size, method, num_heads, merge='mean'))
            

        # Define Readout
        self.r = ReadoutFunction('mpnn',
                                 args={'in': hidden_state_size,
                                       'target': l_target})
        # self.g = g
        # equation (1)
        self.fc = nn.Linear(hidden_state_size, hidden_state_size, bias=False)
        
        self.type = type

        self.args = {}
        self.args['out'] = hidden_state_size
        self.n_layers = n_layers

    def forward(self, g, h_in, e):

        h = []

        # Padding to some larger dimension d
        h_t = torch.cat([h_in, Variable(
            torch.zeros(h_in.size(0), h_in.size(1), self.args['out'] - h_in.size(2)).type_as(h_in.data))], 2)

        h.append(h_t.clone())

        # Layer
        for t in range(0, self.n_layers):
            h_t = self.m[t](h[t], e)
            h_t = (torch.sum(h_in, 2, keepdim = True).expand_as(h_t) > 0).type_as(h_t) * h_t
            h.append(h_t)

        # Readout
        res = self.r.forward(h)

        if self.type == 'classification':
            res = nn.LogSoftmax()(res)
        return res


class MPNNSingleHAttn(nn.Module):
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

    def __init__(self, in_n, hidden_state_size, out_dim,  method=3):
        super(MPNNSingleHAttn, self).__init__()

        # equation (1)

        self.fc = nn.Linear(hidden_state_size, out_dim, bias=False)
        # equation (2)
        self.method_v = method
        if method ==1:
            self.attn_fc = nn.Linear(2 * out_dim, 1, bias=False)
        if method ==2:
            self.attn_fc = nn.Linear(out_dim, 1, bias=False)
        if method ==3:
            self.attn_fc = nn.Linear(out_dim*2+in_n[1], 1, bias=False)
        if method ==4:
            self.attn_fc = nn.Linear(in_n[1], 1, bias=False)
        if method ==5:
            self.attn_fc = nn.Linear(2 * out_dim, 1, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.fc.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_fc.weight, gain=gain)
    
    def forward(self, h0, e):
        if self.method_v == 1:
            return self.method1(h0)
        if self.method_v == 2:
            return self.method2(h0)
        if self.method_v == 3:
            return self.method3(h0, e)
        if self.method_v == 4:
            return self.method4(h0, e)
        if self.method_v == 5:
            return self.method5(h0, e)

    def method1(self, h0):
        h_1 = h0.view(-1, h0.size(2))
        z = self.fc(h_1)
        z = z.view(h0.size(0),h0.size(1),-1)
        z1 = z.repeat(1, z.size(1), 1)
        z2 = z.repeat_interleave(z.size(1), dim = 1)

        zij = torch.cat((z1, z2), dim= 2)
        # print('zij', zij.size())
        eij = F.leaky_relu(self.attn_fc(zij))
        # print('eij', eij.size())
        alpha = F.softmax(eij, dim=1)
        # print('alpha', alpha.size())
        # alpha torch.Size([10, 441, 1])
        alpha= alpha.view(h0.size(0),h0.size(1),h0.size(1),-1)
        z = z[:,:,None,:].expand(h0.size(0),h0.size(1),h0.size(1),-1)
        h_t = torch.sum(alpha * z, dim=2)
        return h_t

    def method2(self, h0):
        h_1 = h0.view(-1, h0.size(2))
        z = self.fc(h_1)
        # print(z.shape)
        z = z.view(h0.size(0),h0.size(1),-1)
        z1 = z.repeat(1, z.size(1), 1)
        z2 = z.repeat_interleave(z.size(1), dim = 1)

        zij = z1*z2
        # print('zij', zij.size())
        eij = F.leaky_relu(self.attn_fc(zij))
        # print('eij', eij.size())
        alpha = F.softmax(eij, dim=1)
        # print('alpha', alpha.size())
        # alpha torch.Size([10, 441, 1])
        alpha= alpha.view(h0.size(0),h0.size(1),h0.size(1),-1)
        z = z[:,:,None,:].expand(h0.size(0),h0.size(1),h0.size(1),-1)
        h_t = torch.sum(alpha * z, dim=2)
        return h_t

    def method3(self, h0, e):
        h_1 = h0.view(-1, h0.size(2))
        z = self.fc(h_1)
        # print(z.shape)
        z = z.view(h0.size(0),h0.size(1),-1)
        z1 = z.repeat(1, z.size(1), 1)
        z2 = z.repeat_interleave(z.size(1), dim = 1)
        m = e.view(e.size(0), -1, e.size(3))
        zij = torch.cat((z1, z2, m), dim= 2)
        eij = F.leaky_relu(self.attn_fc(zij))
        alpha = F.softmax(eij, dim=1)
        alpha= alpha.view(h0.size(0),h0.size(1),h0.size(1),-1)
        z = z[:,:,None,:].expand(h0.size(0),h0.size(1),h0.size(1),-1)
        h_t = torch.sum(alpha * z, dim=2)
        return h_t

    def method4(self, h0, e):
        h_1 = h0.view(-1, h0.size(2))
        z = self.fc(h_1)
        z = z.view(h0.size(0),h0.size(1),-1)
        edij = e.view(e.size(0), -1, e.size(3))
        eij = F.leaky_relu(self.attn_fc(edij))
        alpha = F.softmax(eij, dim=1)
        alpha= alpha.view(h0.size(0),h0.size(1),h0.size(1),-1)
        z = z[:,:,None,:].expand(h0.size(0),h0.size(1),h0.size(1),-1)
        h_t = torch.sum(alpha * z, dim=2)
        return h_t

    def method5(self, h0, e):
        h_1 = h0.view(-1, h0.size(2))
        z = self.fc(h_1)
        z = z.view(h0.size(0),h0.size(1),-1)
        z1 = z.repeat(1, z.size(1), 1)
        z2 = z.repeat_interleave(z.size(1), dim = 1)
        zij = torch.cat((z1, z2), dim= 2)
        eij = self.attn_fc(F.leaky_relu(zij))
        alpha = F.softmax(eij, dim=1)
        alpha= alpha.view(h0.size(0),h0.size(1),h0.size(1),-1)
        z = z[:,:,None,:].expand(h0.size(0),h0.size(1),h0.size(1),-1)
        h_t = torch.sum(alpha * z, dim=2)
        return h_t




class MPNNMultiHAttn(nn.Module):
    """

        Parameters
        ----------
        in_n : int list
            Sizes for the node and edge features.
        hidden_state_size : int
            Size of the hidden states (the input will be padded with 0's to this size).
        
    """

    def __init__(self, in_n, hidden_state_size, out_dim, method=3,num_heads=3, merge= 'mean'):
        super(MPNNMultiHAttn, self).__init__()

        # self.g = g
        # equation (1)
        self.heads = nn.ModuleList()
        for i in range(num_heads):
            self.heads.append(MPNNSingleHAttn(in_n, hidden_state_size, out_dim,  method))
        self.merge = merge


    def forward(self, h_in, e):
        # print(h_in.size())
        if self.merge == 'cat':
            heads_out = [torch.sigmoid(attn_head(h_in, e)) for attn_head in self.heads]
            return torch.cat(heads_out, dim=-1)
        else:
            heads_out = [attn_head(h_in, e) for attn_head in self.heads]
            return torch.sigmoid(torch.mean(torch.stack(heads_out), axis = 0))
            # return torch.mean(torch.stack(heads_out))

