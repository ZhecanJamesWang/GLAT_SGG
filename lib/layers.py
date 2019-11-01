import math

import torch

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pdb
from .layernorm import LayerNormalization
from torch.autograd import Variable

class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        # self.weight = Parameter(torch.DoubleTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        # pdb.set_trace()
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None):

        # (b，n, d) (b, d, n) = (b, n, n)
        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature


        if mask is not None:
            attn = attn.masked_fill(mask, -np.inf)

        attn = self.softmax(attn)
        attn = self.dropout(attn)

        # (b, n, n) (b, n, d) = (b, n, d)
        output = torch.bmm(attn, v)

        return output, attn


class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k)
        self.w_ks = nn.Linear(d_model, n_head * d_k)
        self.w_vs = nn.Linear(d_model, n_head * d_v)

        # nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        # nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        # nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))
        # self.layer_norm = nn.LayerNorm(d_model)
        self.layer_norm = LayerNormalization(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)
        # nn.init.xavier_normal_(self.fc.weight)

        self.dropout = nn.Dropout(dropout)

        self.fc1 = nn.Linear(2 * d_model, d_model)

    def forward(self, q, k, v, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head


        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        # sz_b, len_q = q.size()
        # sz_b, len_k = k.size()
        # sz_b, len_v = v.size()

        residual = q

        # (b, n, d) (d, nhead*k) => (b, n, nhead*k)
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k) # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k) # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v) # (n*b) x lv x dv

        mask = mask.repeat(n_head, 1, 1) # (n*b) x .. x ..
        output, attn = self.attention(q, k, v, mask=mask)
        # output, attn = self.attention(q, k, v)

        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1) # b x lq x (n*dv)


        output = self.dropout(self.fc(output))



        # output = self.layer_norm(output + residual)

        output = torch.cat((output, residual), 2)
        output = self.layer_norm(self.fc1(output))

        return output, attn


class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Conv1d(d_in, d_hid, 1) # position-wise
        self.w_2 = nn.Conv1d(d_hid, d_in, 1) # position-wise
        # self.layer_norm = nn.LayerNorm(d_in)
        self.layer_norm = LayerNormalization(d_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        output = x.transpose(1, 2)
        output = self.w_2(F.relu(self.w_1(output)))
        output = output.transpose(1, 2)
        output = self.dropout(output)
        output = self.layer_norm(output + residual)
        output = output + residual
        return output


class EncoderLayer(nn.Module):
    ''' Compose with two layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(
            n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, enc_input, non_pad_mask=None, slf_attn_mask=None):
        # enc_output, enc_slf_attn = self.slf_attn(
        #     enc_input, enc_input, enc_input)

        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, mask=slf_attn_mask)

        enc_output *= non_pad_mask

        enc_output = self.pos_ffn(enc_output)

        enc_output *= non_pad_mask

        return enc_output, enc_slf_attn


class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()

        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        # nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2*out_features, 1)))
        # nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

        self.fc =  nn.Linear(2 * out_features, out_features)

    # (b, G_n, N_n, N_n) (b, G_n, N_n, d)

    def forward(self, input, adj):
        # pdb.set_trace()


        # input (b, N_n, d)

        # input (b*n, N_n, d) W (d, d) => h (b*n, N_n, d)
        h = torch.matmul(input, self.W)
        # N = h.size()[0]
        B = h.size()[0]
        N = h.size()[1]
        # D = h.size()[2]

        # h(b, N_n, d) => a_input (b, N_n, N_n, 2*d)
        a_input = torch.cat([h.repeat(1, 1, N).view(B, N * N, -1), h.repeat(1, N, 1)], dim=2).view(B, N, -1, 2 * self.out_features)


        # a_input(b, N_n, N_n, 2*d) *  a (2*d, 1) = (b, N_n, N_n, 1) => e (b, N_n, N_n)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(3))

        zero_vec = -9e15*torch.ones_like(e)

        # adj (B, N_n, N_n) attention (b, N_n, N_n) e (b, N_n, N_n) zero vec(b, N_n, N_n)
        attention = torch.where(adj > 0, e, zero_vec)

        # (b * n, N_n, N_n)

        attention = F.softmax(attention, dim=2)
        attention = F.dropout(attention, self.dropout, training=self.training)




        # h_prime = torch.matmul(attention, h) + h

        # attention (B, N_n, N_n) * h (b, N_n, d) => (b, N_n, d)
        h_prime = torch.cat((torch.matmul(attention, h), h), 2)
        h_prime = self.fc(h_prime)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime


        # (b*n, N_n, d) (b*n, d, N_n) = (b*n, N_n, N_n)

        # adj => (b, N_n, N_n) => (b*n, N_n, N_n)


        # a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2 * self.out_features)
        #
        # e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))
        #
        # zero_vec = -9e15*torch.ones_like(e)
        # attention = torch.where(adj > 0, e, zero_vec)
        # attention = F.softmax(attention, dim=1)
        # attention = F.dropout(attention, self.dropout, training=self.training)
        # h_prime = torch.matmul(attention, h)
        #
        # if self.concat:
        #     return F.elu(h_prime)
        # else:
        #     return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'

class GraphAtt_Mutlihead_Basic(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, n_head, d_model, d_k, d_v, foc_type, dropout=0.1):
        super(GraphAtt_Mutlihead_Basic, self).__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k)
        self.w_ks = nn.Linear(d_model, n_head * d_k)
        self.w_vs = nn.Linear(d_model, n_head * d_v)

        self.attention = GraphAtt_ScaledDotProduct(np.power(d_k, 0.5), dropout, foc_type)
        # self.layer_norm = nn.LayerNorm(d_model)
        self.layer_norm = LayerNormalization(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)
        # nn.init.xavier_normal_(self.fc.weight)

        self.dropout = nn.Dropout(dropout)

        self.fc1 = nn.Linear(2 * d_model, d_model)

    def forward(self, q, k, v, adj, mask=None):

        # (b, N_n, d)

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q

        # (b, n, d) (d, nhead*k) => (b, n, nhead*k)
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k) # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k) # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v) # (n*b) x lv x dv

        mask = mask.repeat(n_head, 1, 1) # (n*b) x .. x ..
        output, attn = self.attention(q, k, v, adj, mask=mask)
        # output, attn = self.attention(q, k, v)

        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1) # b x lq x (n*dv)

        output = self.dropout(self.fc(output))

        # output = self.layer_norm(output + residual)

        output = torch.cat((output, residual), 2)
        output = self.layer_norm(self.fc1(output))

        return output, attn

    # def __repr__(self):
    #     return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class GraphAtt_ScaledDotProduct(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, temperature, dropout, foc_type, concat=True):
        super(GraphAtt_ScaledDotProduct, self).__init__()

        self.temperature = temperature
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=2)
        self.foc_type = foc_type

    def forward(self, q, k, v, adj, mask=None):
        a, N_n, _ = v.size()
        b, _, _ = adj.size()
        nhead = int(a/b)

        # (b*n, N_n, d) (b*n, d, N_n) = (b*n, N_n, N_n)
        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature

        if mask is not None:
            # attn = attn.masked_fill(mask, -np.inf)
            attn = attn.masked_fill(mask, -9e6)

        if self.foc_type == "local":
            # adj => (b, N_n, N_n) => (b*n, N_n, N_n)
            adj = adj.unsqueeze(0)
            adj = adj.repeat(nhead, 1, 1, 1)
            adj = adj.view(-1, N_n, N_n)

            zero_vec = -9e15*torch.ones_like(attn)

            attn_mask = adj > 0
            zero_vec_mask = adj <= 0

            attn = attn_mask.float() * attn
            attn += zero_vec_mask.float() * zero_vec

            # attn = torch.where(adj > 0, attn, zero_vec)

        attn = self.softmax(attn)
        attn = self.dropout(attn)

        # attention (b*n, N_n, N_n) v (b*n, N_n, d) => (b*n, N_n, d)
        output = torch.bmm(attn, v)

        return output, attn
