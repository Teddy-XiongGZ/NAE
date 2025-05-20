import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.parameter import Parameter
import math

class MultiChannelLinear(nn.Module):
    
    def __init__(self, in_dim, out_dim, n_channel=1, reduction="sum"):
        super(MultiChannelLinear, self).__init__()
        
        self.reduction = reduction

        #initialize weights
        self.w = torch.nn.Parameter(torch.zeros(n_channel, out_dim, in_dim))
        self.b = torch.nn.Parameter(torch.zeros(1, n_channel, out_dim))
        
        #change weights to kaiming
        # nn.init.kaiming_uniform_(self.w, a=math.sqrt(3))
        nn.init.kaiming_uniform_(self.w, nonlinearity="relu")
        # fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.w)
        fan_in = n_channel * in_dim
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.b, -bound, bound)
    
    def forward(self, x):
        '''
            args:
                x: input, whose shape can be 
                    batch_size, (channel), in_dim
            return:
                output, whose shape will be
                    batch_size, (channel), out_dim
        '''
        # b, ch, r, c  = x.size()
        # return (( x * self.w).sum(-1 ) + self.b).view(b,ch,1,-1)
        if self.reduction == "sum":
            return (self.w * x.unsqueeze(-2)).sum(-1) + self.b
        elif self.reduction == "mean":
            return (self.w * x.unsqueeze(-2)).mean(-1) + self.b


class Adaptive_NAM(nn.Module):

    def __init__(self, problem, n_feat, h_dim, n_layers, n_class, C=4, k=4, dropout=0.0, dropout_expert=0.0, output_penalty=0.0, var_penalty=0.0, batch_norm=False):
        super(Adaptive_NAM, self).__init__()

        self.problem = problem
        self.n_feat = n_feat
        self.h_dim = h_dim
        self.n_class = n_class
        self.n_layers = n_layers
        self.C = C
        self.k = k

        self.output_penalty = output_penalty
        self.var_penalty = var_penalty

        assert self.n_layers > 0

        self.dropout = nn.Dropout(p=dropout)
        self.dropout_expert = nn.Dropout(p=dropout_expert)
        self.activate = nn.ReLU()

        if batch_norm:
            self.norm = nn.BatchNorm1d(self.n_feat)
        else:
            self.norm = nn.LayerNorm(self.h_dim)

        self.enc_list = nn.Sequential(
            nn.Sequential(MultiChannelLinear(1, self.h_dim, self.n_feat), self.norm, self.activate, self.dropout),
            *([nn.Sequential(
                MultiChannelLinear(self.h_dim, self.h_dim, self.n_feat), self.norm, self.activate, self.dropout
            ) for _ in range(self.n_layers - 1)])
        )

        self.experts = MultiChannelLinear(self.h_dim, self.C, self.n_feat)

        self.router = nn.Linear(self.n_feat * self.h_dim, self.n_feat * self.C)
        self.bias = nn.parameter.Parameter(torch.zeros(1, self.n_class), requires_grad = True)

        if self.problem == "regression":
            self.criterion = nn.MSELoss()
        elif self.n_class == 1:
            self.criterion = nn.BCEWithLogitsLoss()
        else:
            self.criterion = nn.CrossEntropyLoss()

        nn.init.xavier_normal_(self.router.weight)

    def forward(self, x, y):
        '''
        Input:
            x: input features (batch_size, n_feat)
            y: target label / value (batch_size)
        '''
        bsz = len(y)
        Z = self.encode(x) # (batch_size, n_feat, h_dim)
        res, output_loss = self.predict(Z)
        total_loss = self.criterion(res, y)
        return total_loss + output_loss, res
    
    def encode(self, x):
        Z = self.enc_list(x.unsqueeze(-1)) # (batch_size, n_feat, h_dim)
        return Z

    def predict(self, Z):
        res = self.experts(Z) # (batch_size, n_feat, C)
        loss = res.var(dim=-1).mean() * self.var_penalty
        prob = self.router(Z.flatten(start_dim=-2)).view(-1, self.n_feat, self.C) # (batch_size, n_feat, C)
        masks = - torch.ones_like(prob) * np.inf
        k = self.k
        mask_indices = prob.argsort(dim=-1, descending=True)[:, :, :k]
        batch_indices = torch.arange(prob.shape[0]).unsqueeze(1).unsqueeze(2).expand(-1, prob.shape[1], k)
        sequence_indices = torch.arange(prob.shape[1]).unsqueeze(0).unsqueeze(2).expand(prob.shape[0], -1, k)
        masks[batch_indices, sequence_indices, mask_indices] = 0
        weights = torch.softmax(prob + masks, dim=-1) # (batch_size, n_feat, C)
        weights = self.dropout_expert(weights)
        res = res * weights # (batch_size, n_feat, C)
        res = res.sum(dim=-1) # (batch_size, n_feat)
        loss += res.pow(2).mean() * self.output_penalty
        res = res.sum(dim=-1, keepdim=True) + self.bias # (batch_size)
        return res.squeeze(-1), loss

class Adaptive_NAM_D(nn.Module):

    def __init__(self, problem, n_feat, h_dim, n_layers, n_class, C=4, k=4, dropout=0.0, dropout_expert=0.0, output_penalty=0.0, var_penalty=0.0, batch_norm=False):
        super(Adaptive_NAM_D, self).__init__()

        self.problem = problem
        self.n_feat = n_feat
        self.h_dim = h_dim
        self.n_class = n_class
        self.n_layers = n_layers
        self.C = C
        self.k = k

        self.output_penalty = output_penalty
        self.var_penalty = var_penalty

        assert self.n_layers > 0

        self.dropout = nn.Dropout(p=dropout)
        self.dropout_expert = nn.Dropout(p=dropout_expert)
        self.activate = nn.ReLU()

        if batch_norm:
            self.norm = nn.BatchNorm1d(self.n_feat)
        else:
            self.norm = nn.LayerNorm(self.h_dim)

        self.enc_list = nn.Sequential(
            nn.Sequential(MultiChannelLinear(1, self.h_dim, self.n_feat), self.norm, self.activate, self.dropout),
            *([nn.Sequential(
                MultiChannelLinear(self.h_dim, self.h_dim, self.n_feat), self.norm, self.activate, self.dropout
            ) for _ in range(self.n_layers - 1)])
        )

        self.experts = MultiChannelLinear(self.h_dim, self.C, self.n_feat)

        self.router = MultiChannelLinear(self.h_dim, self.C, self.n_feat)
        self.bias = nn.parameter.Parameter(torch.zeros(1, self.n_class), requires_grad = True)

        if self.problem == "regression":
            self.criterion = nn.MSELoss()
        elif self.n_class == 1:
            self.criterion = nn.BCEWithLogitsLoss()
        else:
            self.criterion = nn.CrossEntropyLoss()
    
    def forward(self, x, y):
        '''
        Input:
            x: input features (batch_size, n_feat)
            y: target label / value (batch_size)
        '''
        bsz = len(y)
        Z = self.encode(x) # (batch_size, n_feat, h_dim)
        res, output_loss = self.predict(Z)
        total_loss = self.criterion(res, y)
        return total_loss + output_loss, res
    
    def encode(self, x):
        Z = self.enc_list(x.unsqueeze(-1)) # (batch_size, n_feat, h_dim)
        return Z

    def predict(self, Z):
        res = self.experts(Z) # (batch_size, n_feat, C)
        loss = res.var(dim=-1).mean() * self.var_penalty
        prob = self.router(Z).view(-1, self.n_feat, self.C) # (batch_size, n_feat, C)
        temperature = 0.1
        prob = torch.log(torch.softmax(prob, dim=-1))
        G = -torch.log(-torch.log(torch.rand_like(prob)))
        weights = torch.softmax((prob + G) / temperature, dim=-1)
        weights = self.dropout_expert(weights)
        res = res * weights # (batch_size, n_feat, C)
        res = res.sum(dim=-1) # (batch_size, n_feat)
        loss += res.pow(2).mean() * self.output_penalty
        res = res.sum(dim=-1, keepdim=True) + self.bias # (batch_size)
        return res.squeeze(-1), loss


class Adaptive_NAM_E(nn.Module):

    def __init__(self, problem, n_feat, h_dim, n_layers, n_class, C=4, k=4, dropout=0.0, dropout_expert=0.0, output_penalty=0.0, var_penalty=0.0, batch_norm=False):
        super(Adaptive_NAM_E, self).__init__()

        self.problem = problem
        self.n_feat = n_feat
        self.h_dim = h_dim
        self.n_class = n_class
        self.n_layers = n_layers
        self.C = C
        self.k = k

        self.output_penalty = output_penalty
        self.var_penalty = var_penalty

        assert self.n_layers > 0

        self.dropout = nn.Dropout(p=dropout)
        self.dropout_expert = nn.Dropout(p=dropout_expert)
        self.activate = nn.ReLU()

        if batch_norm:
            self.norm = nn.BatchNorm1d(self.n_feat)
        else:
            self.norm = nn.LayerNorm(self.h_dim)

        self.enc_list = nn.Sequential(
            nn.Sequential(MultiChannelLinear(1, self.h_dim, self.n_feat), self.norm, self.activate, self.dropout),
            *([nn.Sequential(
                MultiChannelLinear(self.h_dim, self.h_dim, self.n_feat), self.norm, self.activate, self.dropout
            ) for _ in range(self.n_layers - 1)])
        )

        self.experts = MultiChannelLinear(self.h_dim, self.C, self.n_feat)

        self.router = nn.Linear(self.n_feat * self.h_dim, self.n_feat * self.C)
        self.bias = nn.parameter.Parameter(torch.zeros(1, self.n_class), requires_grad = True)

        if self.problem == "regression":
            self.criterion = nn.MSELoss()
        elif self.n_class == 1:
            self.criterion = nn.BCEWithLogitsLoss()
        else:
            self.criterion = nn.CrossEntropyLoss()

        nn.init.xavier_normal_(self.router.weight)
        
    def forward(self, x, y):
        '''
        Input:
            x: input features (batch_size, n_feat)
            y: target label / value (batch_size)
        '''
        bsz = len(y)
        Z = self.encode(x) # (batch_size, n_feat, h_dim)
        res, output_loss = self.predict(Z)
        total_loss = self.criterion(res, y)
        return total_loss + output_loss, res
    
    def encode(self, x):
        Z = self.enc_list(x.unsqueeze(-1)) # (batch_size, n_feat, h_dim)
        return Z

    def predict(self, Z):
        res = self.experts(Z) # (batch_size, n_feat, C)
        loss = res.var(dim=-1).mean() * self.var_penalty
        prob = self.router(Z.flatten(start_dim=-2)).view(-1, self.n_feat, self.C) # (batch_size, n_feat, C)
        masks = - torch.ones_like(prob) * np.inf
        k = self.k
        mask_indices = prob.argsort(dim=-1, descending=True)[:, :, :k]
        batch_indices = torch.arange(prob.shape[0]).unsqueeze(1).unsqueeze(2).expand(-1, prob.shape[1], k)
        sequence_indices = torch.arange(prob.shape[1]).unsqueeze(0).unsqueeze(2).expand(prob.shape[0], -1, k)
        masks[batch_indices, sequence_indices, mask_indices] = 0
        weights = torch.softmax(prob - prob.data + masks, dim=-1) # (batch_size, n_feat, C)
        weights = self.dropout_expert(weights)
        res = res * weights # (batch_size, n_feat, C)
        res = res.sum(dim=-1) # (batch_size, n_feat)
        loss += res.pow(2).mean() * self.output_penalty
        res = res.sum(dim=-1, keepdim=True) + self.bias # (batch_size)
        return res.squeeze(-1), loss
