import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import math
from scipy.stats import norm
import scipy
from collections import OrderedDict

import sys
sys.path.append("net/Temporal_shift")
from cuda.shift import Shift

def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


def conv_init(conv):
    nn.init.kaiming_normal(conv.weight, mode='fan_out')
    nn.init.constant(conv.bias, 0)

def bn_init(bn, scale):
    nn.init.constant(bn.weight, scale)
    nn.init.constant(bn.bias, 0)


class tcn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1):
        super(tcn, self).__init__()
        pad = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), padding=(pad, 0),
                              stride=(stride, 1))

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        conv_init(self.conv)
        bn_init(self.bn, 1)

    def forward(self, x):
        x = self.bn(self.conv(x))
        return x


class Shift_tcn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1):
        super(Shift_tcn, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.bn = nn.BatchNorm2d(in_channels)
        self.bn2 = nn.BatchNorm2d(in_channels)
        bn_init(self.bn2, 1)
        self.relu = nn.ReLU(inplace=True)
        self.shift_in = Shift(channel=in_channels, stride=1, init_scale=1)
        self.shift_out = Shift(channel=out_channels, stride=stride, init_scale=1)

        self.temporal_linear = nn.Conv2d(in_channels, out_channels, 1)
        nn.init.kaiming_normal(self.temporal_linear.weight, mode='fan_out')

    def forward(self, x):
        x = self.bn(x)
        # shift1
        x = self.shift_in(x)
        x = self.temporal_linear(x)
        x = self.relu(x)
        # shift2
        x = self.shift_out(x)
        x = self.bn2(x)
        return x


class Shift_gcn(nn.Module):
    def __init__(self, in_channels, out_channels, A, coff_embedding=4, num_subset=3):
        super(Shift_gcn, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if in_channels != out_channels:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.down = lambda x: x

        self.Linear_weight = nn.Parameter(torch.zeros(in_channels, out_channels, requires_grad=True, device='cuda'),
                                          requires_grad=True)
        nn.init.normal_(self.Linear_weight, 0, math.sqrt(1.0 / out_channels))

        self.Linear_bias = nn.Parameter(torch.zeros(1, 1, out_channels, requires_grad=True, device='cuda'),
                                        requires_grad=True)
        nn.init.constant(self.Linear_bias, 0)

        self.Feature_Mask = nn.Parameter(torch.ones(1, 25, in_channels, requires_grad=True, device='cuda'),
                                         requires_grad=True)
        nn.init.constant(self.Feature_Mask, 0)

        self.bn = nn.BatchNorm1d(25 * out_channels)
        self.relu = nn.ReLU()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)

        index_array = np.empty(25 * in_channels).astype(int)
        # index_array = np.empty(25*in_channels, dtype=int)
        for i in range(25):
            for j in range(in_channels):
                index_array[i * in_channels + j] = (i * in_channels + j + j * in_channels) % (in_channels * 25)
        self.shift_in = nn.Parameter(torch.from_numpy(index_array), requires_grad=False)

        index_array = np.empty(25 * out_channels).astype(int)
        for i in range(25):
            for j in range(out_channels):
                index_array[i * out_channels + j] = (i * out_channels + j - j * out_channels) % (out_channels * 25)
        self.shift_out = nn.Parameter(torch.from_numpy(index_array), requires_grad=False)

    def forward(self, x0):
        n, c, t, v = x0.size()
        x = x0.permute(0, 2, 3, 1).contiguous()

        # shift1
        x = x.view(n * t, v * c)
        x = torch.index_select(x, 1, self.shift_in)
        x = x.view(n * t, v, c)
        x = x * (torch.tanh(self.Feature_Mask) + 1)

        x = torch.einsum('nwc,cd->nwd', (x, self.Linear_weight)).contiguous()  # nt,v,c
        x = x + self.Linear_bias

        # shift2
        x = x.view(n * t, -1)
        x = torch.index_select(x, 1, self.shift_out)
        x = self.bn(x)
        x = x.view(n, t, v, self.out_channels).permute(0, 3, 1, 2)  # n,c,t,v

        x = x + self.down(x0)
        x = self.relu(x)
        return x


class TCN_GCN_unit(nn.Module):
    def __init__(self, in_channels, out_channels, A, stride=1, residual=True):
        super(TCN_GCN_unit, self).__init__()
        self.gcn1 = Shift_gcn(in_channels, out_channels, A)
        self.tcn1 = Shift_tcn(out_channels, out_channels, stride=stride)
        self.relu = nn.ReLU()

        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = tcn(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x):
        x = self.tcn1(self.gcn1(x)) + self.residual(x)
        return self.relu(x)


class Model(nn.Module):
    def __init__(self, num_class=60, num_point=25, num_person=2, graph=None, graph_args=dict(), in_channels=3,
                 return_ft=False):
        super(Model, self).__init__()

        if graph is None:
            raise ValueError()
        else:
            Graph = import_class(graph)
            self.graph = Graph(**graph_args)

        A = self.graph.A
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)

        self.l1 = TCN_GCN_unit(3, 64, A, residual=False)
        self.l2 = TCN_GCN_unit(64, 64, A)
        self.l3 = TCN_GCN_unit(64, 64, A)
        self.l4 = TCN_GCN_unit(64, 64, A)
        self.l5 = TCN_GCN_unit(64, 128, A, stride=2)
        self.l6 = TCN_GCN_unit(128, 128, A)
        self.l7 = TCN_GCN_unit(128, 128, A)
        self.l8 = TCN_GCN_unit(128, 256, A, stride=2)
        self.l9 = TCN_GCN_unit(256, 256, A)
        self.l10 = TCN_GCN_unit(256, 256, A)

        self.fc = nn.Linear(256, num_class)
        nn.init.normal(self.fc.weight, 0, math.sqrt(2. / num_class))
        bn_init(self.data_bn, 1)
        self.return_ft = return_ft

    def forward(self, x):
        N, C, T, V, M = x.size()

        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)

        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        x = self.l6(x)
        x = self.l7(x)
        x = self.l8(x)
        x = self.l9(x)
        x = self.l10(x)

        # N*M,C,T,V
        c_new = x.size(1)

        # x_match_feature = x.view(N, c_new, T // 4, V, M)
        # x_match_feature = x_match_feature.mean(4)

        x = x.view(N, M, c_new, -1)
        x = x.mean(3).mean(1)
        if self.return_ft:
            return self.fc(x), x
        else:
            x = self.fc(x)
            return x
        # return x, self.fc(x)
        # return x_match_feature, x


def edge2mat(link, num_node):
    A = np.zeros((num_node, num_node))
    for i, j in link:
        A[j, i] = 1
    return A


def get_k_scale_graph(scale, A):
    if scale == 1:
        return A
    An = np.zeros_like(A)
    A_power = np.eye(A.shape[0])
    for k in range(scale):
        A_power = A_power @ A
        An += A_power
    An[An > 0] = 1
    return An


def normalize_digraph(A):
    Dl = np.sum(A, 0)
    h, w = A.shape
    Dn = np.zeros((w, w))
    for i in range(w):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i] ** (-1)
    AD = np.dot(A, Dn)
    return AD

def get_spatial_graph(num_node, self_link, inward, outward):
    I = edge2mat(self_link, num_node)
    In = normalize_digraph(edge2mat(inward, num_node))
    Out = normalize_digraph(edge2mat(outward, num_node))
    A = np.stack((I, In, Out))
    return A


def normalize_adjacency_matrix(A):
    node_degrees = A.sum(-1)
    degs_inv_sqrt = np.power(node_degrees, -0.5)
    norm_degs_matrix = np.eye(len(node_degrees)) * degs_inv_sqrt
    return (norm_degs_matrix @ A @ norm_degs_matrix).astype(np.float32)


def get_uniform_graph(num_node, self_link, neighbor):
    A = normalize_digraph(edge2mat(neighbor + self_link, num_node))
    return A


# num_node = 25
# self_link = [(i, i) for i in range(num_node)]
# inward_ori_index = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6),
#                     (8, 7), (9, 21), (10, 9), (11, 10), (12, 11), (13, 1),
#                     (14, 13), (15, 14), (16, 15), (17, 1), (18, 17), (19, 18),
#                     (20, 19), (22, 23), (23, 8), (24, 25), (25, 12)]
# inward = [(i - 1, j - 1) for (i, j) in inward_ori_index]
# outward = [(j, i) for (i, j) in inward]
# neighbor = inward + outward


class Graph:
    def __init__(self, labeling_mode='spatial', type='ntu'):
        if type == 'ntu':
            self.num_node = 25
            self.self_link = [(i, i) for i in range(25)]
            inward_ori_index = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6),
                                (8, 7), (9, 21), (10, 9), (11, 10), (12, 11), (13, 1),
                                (14, 13), (15, 14), (16, 15), (17, 1), (18, 17), (19, 18),
                                (20, 19), (22, 23), (23, 8), (24, 25), (25, 12)]
            inward = [(i - 1, j - 1) for (i, j) in inward_ori_index]
            outward = [(j, i) for (i, j) in inward]
            neighbor = inward + outward
            self.inward = inward
            self.outward = outward
            self.neighbor = neighbor
        elif type == 'sbu':
            self.num_node = 15
            self.self_link = [(i, i) for i in range(15)]
            inward_ori_index = [(1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (3, 7),
                             (7, 8), (8, 9), (3, 10), (10, 11), (11, 12),
                             (3, 13), (13, 14), (14, 15)]
            inward = [(i - 1, j - 1) for (i, j) in inward_ori_index]
            outward = [(j, i) for (i, j) in inward]
            neighbor = inward + outward
            self.inward = inward
            self.outward = outward
            self.neighbor = neighbor
        elif type == 'orgbd' or type == 'msdra':
            self.num_node = 20
            self.self_link = [(i, i) for i in range(20)]
            inward_ori_index = [
                (4, 3), (3, 2), (2, 1),
                (3, 5), (5, 6), (6, 7), (7, 8),
                (3, 9), (9, 10), (10, 11), (11, 12),
                (1, 17), (17, 18), (18, 19), (19, 20),
                (1, 13), (13, 14), (14, 15), (15, 16),
            ]
            inward = [(i - 1, j - 1) for (i, j) in inward_ori_index]
            outward = [(j, i) for (i, j) in inward]
            neighbor = inward + outward
            self.inward = inward
            self.outward = outward
            self.neighbor = neighbor
        elif type == 'posetics':
            self.num_node = 17
            self.self_link = [(i, i) for i in range(17)]
            inward_ori_index = [
                (0, 1), (1, 2), (2, 3), (2, 5), (2, 6),
                (5, 7), (7, 9), (6, 8), (8, 10), (3, 4),
                (4, 11), (4, 12),
                (11, 13), (13, 15), (12, 14), (14, 16)
            ]
            inward = [(i, j) for (i, j) in inward_ori_index]
            outward = [(j, i) for (i, j) in inward]
            neighbor = inward + outward
            self.inward = inward
            self.outward = outward
            self.neighbor = neighbor
        self.A = self.get_adjacency_matrix(labeling_mode)

    def get_adjacency_matrix(self, labeling_mode=None):
        if labeling_mode is None:
            return self.A
        if labeling_mode == 'spatial':
            A = get_spatial_graph(self.num_node, self.self_link, self.inward, self.outward)
        else:
            raise ValueError()
        return A
