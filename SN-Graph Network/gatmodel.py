import os.path as osp
from math import ceil, pi
import math
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, EdgeConv
from torch_geometric.utils import to_dense_batch, to_dense_adj, subgraph, add_self_loops
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp,\
     global_sort_pool as gsp, radius
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, BatchNorm1d as BN, Dropout
from torch_geometric.transforms import LinearTransformation

def MLP(channels, batch_norm=True):
    return Seq(*[
        Seq(Lin(channels[i - 1], channels[i]), 
        BN(channels[i]), 
        ReLU())
        for i in range(1, len(channels))
    ])

def random_drop_node(node_num, batch_size, p, p2 = 1):
    mask = None
    p = random.uniform(p, p2)
    for batch in range(batch_size):
        _mask = np.ones((node_num), dtype=bool)
        id = np.arange(0, node_num)
        np.random.shuffle(id)
        get_num = (int)((node_num) * (1 - p))
        for index in range(get_num):
            _mask[id[index]] = False
        if(mask is None):
            mask = _mask
        else:
            mask = np.concatenate((mask, _mask))
    return mask, torch.tensor(mask)

def random_rotate(xyz, rotate_type):
    if(rotate_type != 'z' and rotate_type != 'xyz'):
        return xyz
    pi = 3.1415926535
    if(rotate_type == 'xyz'):
        rotateAngleA = random.random() * pi * 2
        rotateAngleB = random.random() * pi * 2
        rotateAngleC = random.random() * pi * 2
        sinA, cosA = sin(rotateAngleA), cos(rotateAngleA)
        sinB, cosB = sin(rotateAngleB), cos(rotateAngleB)
        sinC, cosC = sin(rotateAngleC), cos(rotateAngleC)
        rotation_matrix = torch.tensor([[cosC*cosB, -sinC*cosA+cosC*sinB*sinA, sinC*sinA+cosC*sinB*cosA], 
                [sinC*cosB, cosC*cosA+sinC*sinB*sinA, -cosC*sinA+sinC*sinB*cosA], 
                [-sinB, cosB*sinA, cosB*cosA]]).to(xyz.dtype).to(xyz.device)
        xyz = torch.matmul(xyz, rotation_matrix)
        return xyz
    else:
        rotateAngleA = random.random() * pi * 2
        sinA, cosA = sin(rotateAngleA), cos(rotateAngleA)
        rotation_matrix = torch.tensor([[cosA, 0, sinA],
                                    [0, 1, 0],
                                    [-sinA, 0, cosA]]).to(xyz.dtype).to(xyz.device)
        xyz = torch.matmul(xyz, rotation_matrix)
        return xyz

class NetGAT(torch.nn.Module):
    def __init__(self, node_size, input_feature, num_classes):
        super(NetGAT, self).__init__()
        self.node_per_graph = node_size

        hidden_size = 256
        gat_head = 8
        head_size = hidden_size // gat_head
        self.input_feature = input_feature

        # self.linprev = MLP([input_feature, 64, 64, 64])
        self.linprev = EdgeConv(MLP([input_feature*2, 64, 64, 64]), aggr = 'max')

        self.conv1 = GATConv(64, head_size, gat_head)
        self.bn1 = torch.nn.BatchNorm1d(hidden_size)
        self.lin1 = torch.nn.Linear(64, hidden_size)

        self.conv2 = GATConv(hidden_size, head_size, gat_head)
        self.bn2 = torch.nn.BatchNorm1d(hidden_size)
        self.lin2 = torch.nn.Linear(hidden_size, hidden_size)

        self.conv3 = GATConv(hidden_size, head_size, gat_head)
        self.bn3 = torch.nn.BatchNorm1d(hidden_size)
        self.lin3 = torch.nn.Linear(hidden_size, hidden_size)

        self.conv4 = GATConv(hidden_size, head_size, gat_head)
        self.bn4 = torch.nn.BatchNorm1d(hidden_size)
        self.lin4 = torch.nn.Linear(hidden_size, hidden_size)

        self.mlp = Seq(
            Lin(2048, 512), Dropout(0.4),
            Lin(512, 256), Dropout(0.4),
            Lin(256, num_classes))

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x_posr = x
        batch_size = (int)(batch.size()[0] / self.node_per_graph)

        ##if(self.training == True and random.random()>0.8):
        #    rotateAngleA = random.random() * pi
        #    rotateAngleB = random.random() * pi
        #    rotateAngleC = random.random() * pi
        #    sinA, cosA = math.sin(rotateAngleA), math.cos(rotateAngleA)
        #    sinB, cosB = math.sin(rotateAngleB), math.cos(rotateAngleB)
        #    sinC, cosC = math.sin(rotateAngleC), math.cos(rotateAngleC)
        #    matrix = [[cosC*cosB, -sinC*cosA+cosC*sinB*sinA, sinC*sinA+cosC*sinB*cosA], 
        #              [sinC*cosB, cosC*cosA+sinC*sinB*sinA, -cosC*sinA+sinC*sinB*cosA], 
        #              [-sinB, cosB*sinA, cosB*cosA]]
        #    x_xyz = x[:,0:3]
        #    x_xyz = torch.matmul(x_xyz, torch.tensor(matrix).to(x_xyz.dtype).to(x_xyz.device))
        #    #x_xyz = LinearTransformation(torch.tensor(matrix))(x_xyz)
        #    x_r = x[:,3]
        #    x_r = x_r.reshape((x_r.shape[0], 1))
        #    x = torch.cat((x_xyz, x_r), dim=1)


        add_self_loops(edge_index)
        
        if(self.training == True):
            mask, torchmask = random_drop_node(self.node_per_graph, (int)(batch.size()[0] / self.node_per_graph), 
                                                0.50, 0.50)
            x = x[mask]
            x_posr = x_posr[mask]
            batch = batch[mask]
            edge_index, _ = subgraph(torchmask, edge_index, relabel_nodes = True)
        x0 = self.linprev(x, edge_index)
        x1 = self.conv1(x0, edge_index) + self.lin1(x0)
        x1n = F.relu(x1)
        x2 = self.conv2(x1n, edge_index) + self.lin2(x1n)
        x2n = F.relu(x2)
        x3 = self.conv3(x2n, edge_index) + self.lin3(x2n)
        x3n = F.relu(x3)
        x4 = self.conv4(x3n, edge_index) + self.lin4(x3n)
        x = torch.cat((x1,x2,x3,x4),dim=1)
        x = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        x = self.mlp(x)
        x = F.log_softmax(x, dim = -1)
        return x
