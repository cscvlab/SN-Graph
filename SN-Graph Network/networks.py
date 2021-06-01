import os.path as osp
from math import ceil, sin, cos
import numpy as np
import random

import torch
import torch.nn.functional as F
from torch_geometric.nn import GINConv, EdgeConv, DynamicEdgeConv, GatedGraphConv
from torch_geometric.nn import GraphConv, TopKPooling, PointConv
from torch_geometric.utils import to_dense_batch, to_dense_adj, add_self_loops, dropout_adj, subgraph
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, ELU, BatchNorm1d as BN, Dropout, Conv1d as C1d, MaxPool1d

def MLP(channels, batch_norm=True):
    return Seq(*[
        Seq(Lin(channels[i - 1], channels[i]), 
        BN(channels[i]), 
        ReLU())
        for i in range(1, len(channels))
    ])

class NetGIN(torch.nn.Module):
    def __init__(self, node_per_graph):
        super(Net, self).__init__()

        self.node_per_graph = node_per_graph

        self.linprev = EdgeConv(MLP([4*2, 64, 64, 64]), aggr = 'max')

        self.conv1 = GINConv(MLP([64, 64]))
        self.bn1 = torch.nn.BatchNorm1d(64)
        self.conv2 = GINConv(MLP([64, 128]))
        self.bn2 = torch.nn.BatchNorm1d(128)
        self.conv3 = GINConv(MLP([128, 256]))
        self.bn3 = torch.nn.BatchNorm1d(256)
        self.lin = MLP([512, 1024])

        self.mlp = Seq(
            MLP([2048, 512]), Dropout(0.5), MLP([512, 256]), Dropout(0.5),
            Lin(256, 40))

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        batch_size = (int)(batch.size()[0] / self.node_per_graph)
        
        if(self.training == True):
            mask, torchmask = random_drop_node(self.node_per_graph, (int)(batch.size()[0] / self.node_per_graph), 
                                                0.50, 0.50)
            x = x[mask]
            batch = batch[mask]
            edge_index, _ = subgraph(torchmask, edge_index, relabel_nodes = True)

        x0 = self.linprev(x, edge_index)

        x1 = self.bn1(F.relu(self.conv1(x0, edge_index)))
        x2 = self.bn2(F.relu(self.conv2(x1, edge_index)))
        x3 = self.bn3(F.relu(self.conv3(x2, edge_index)))

        x = torch.cat((x0,x1,x2,x3),dim=1)
        out = self.lin(x)
        
        out = torch.cat([gmp(out, batch), gap(out, batch)], dim = 1)
        out = self.mlp(out)
        out = F.log_softmax(out, dim=-1)

        return out

class GConv(torch.nn.Module):
    def __init__(self, node_size, input_feature):
        super(GConv, self).__init__()

        self.node_per_graph = node_size

        self.linprev = EdgeConv(MLP([input_feature*2, 64, 64, 64]), aggr = 'max')

        self.conv1 = GraphConv(64, 64)
        self.bn1 = torch.nn.BatchNorm1d(64)
        self.conv2 = GraphConv(64, 128)
        self.bn2 = torch.nn.BatchNorm1d(128)
        self.conv3 = GraphConv(128, 256)
        self.bn3 = torch.nn.BatchNorm1d(256)
        self.conv4 = GraphConv(256, 512)
        self.bn4 = torch.nn.BatchNorm1d(512)

        self.mlp = Seq(
            MLP([2048, 512]), Dropout(0.5), MLP([512, 128]), Dropout(0.5),
            Lin(128, 40))

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        if(self.training == True):
            mask, torchmask = random_drop_node(self.node_per_graph, (int)(batch.size()[0] / self.node_per_graph), 0.5, 0.5)
            x = x[mask]
            batch = batch[mask]
            edge_index, _ = subgraph(torchmask, edge_index, relabel_nodes = True)

        x0 = self.linprev(x, edge_index)

        x1 = F.relu(self.conv1(x0, edge_index))
        x1 = self.bn1(x1)
        x2 = F.relu(self.conv2(x1, edge_index))
        x2 = self.bn2(x2)
        x3 = F.relu(self.conv3(x2, edge_index))
        x3 = self.bn3(x3)
        x4 = F.relu(self.conv4(x3, edge_index))
        x4 = self.bn4(x4)

        out = torch.cat([x0,x1,x2,x3,x4], dim=1)
        out = torch.cat([gmp(out, batch), gap(out, batch)], dim = 1)
        out = self.mlp(out)
        out = F.log_softmax(out, dim=-1)

        return out

class GatedGCN(torch.nn.Module):
    def __init__(self, node_size):
        super(GatedGCN, self).__init__()

        self.node_per_graph = node_size

        self.linprev = EdgeConv(MLP([4*2, 64, 64, 64]), aggr = 'max')
        self.conv1 = GatedGraphConv(256, 2)
        self.bn1 = torch.nn.BatchNorm1d(256)
        self.conv2 = GatedGraphConv(256, 2)
        self.bn2 = torch.nn.BatchNorm1d(256)
        self.conv3 = GatedGraphConv(256, 2)
        self.bn3 = torch.nn.BatchNorm1d(256)
        self.conv4 = GatedGraphConv(256, 3)
        self.bn4 = torch.nn.BatchNorm1d(256)

        self.mlp = Seq(
            MLP([256*2*4, 512]), Dropout(0.4), MLP([512, 128]), Dropout(0.4),
            Lin(128, 40))
        

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        if(self.training == True):
            mask, torchmask = random_drop_node(self.node_per_graph, (int)(batch.size()[0] / self.node_per_graph), 0.75, 0.9)
            x = x[mask]
            batch = batch[mask]
            edge_index, _ = subgraph(torchmask, edge_index, relabel_nodes = True)

        x = self.linprev(x, edge_index)

        x = F.relu(self.conv1(x, edge_index))
        x = self.bn1(x)
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        x = F.relu(self.conv2(x, edge_index))
        x = self.bn2(x)
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        x = F.relu(self.conv3(x, edge_index))
        x = self.bn3(x)
        x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        x = F.relu(self.conv4(x, edge_index))
        x = self.bn3(x)
        x4 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = torch.cat([x1,x2,x3,x4], dim=1)
        x = self.mlp(x)
        x = F.log_softmax(x, dim=-1)

        return x