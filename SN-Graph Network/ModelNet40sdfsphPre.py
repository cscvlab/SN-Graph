import os
import os.path as osp
import shutil
from glob import glob

from tqdm import tqdm

import torch
from torch_geometric.data import Dataset, Data, InMemoryDataset
import numpy as np
import math

class ModelNetSdfSpherePre(InMemoryDataset):
    def __init__(self, root, type = "gcn", dataset_type = "40", explict_feature = False, to_1_regularize = True,
                 need_r = True, no_rotate = False, transform = None, pre_transform = None):
        self.root = root
        self.type = type
        self.explict_feature = explict_feature
        self.to_1_regularize = to_1_regularize
        self.need_r = need_r
        self.dataset_type = dataset_type
        #self.num_classes = 40
        #self.num_features = 4

        if(no_rotate):
            self.read_func = self.read_graph_norotate
        else:
            self.read_func = self.read_graph

        super(ModelNetSdfSpherePre, self).__init__(root, transform, pre_transform)

        self.train_dataset = torch.load(self.processed_paths[0])
        self.test_dataset   = torch.load(self.processed_paths[1])

    def process(self):
        files = []
        datas = []
        self.label_name = []
        MN10 = ['bathtub', 'bed', 'chair', 'desk', 'dresser', 'monitor',
            'night_stand', 'sofa', 'table', 'toilet']
        graph_type = 0
        folders = [f.path for f in os.scandir(self.root) if f.is_dir()]
        for folder in folders:
            folder_name = folder.split("/")[-1]
            if(folder_name == "processed"):
                continue
            self.label_name.append(folder_name)
            if(self.dataset_type == "10" and not(folder_name in MN10)):
                continue
            files = glob(osp.join(folder, "train", "*.txt"))
            for file in tqdm(files, ascii=True):
            #for file in files:
                data = self.read_func(file, graph_type)
                datas.append(data)
            graph_type += 1
        self.train_dataset = datas

        datas2 = []
        graph_type2 = 0
        for folder in folders:
            folder_name = folder.split("/")[-1]
            if(folder_name == "processed"):
                continue
            if(self.dataset_type == "10" and not(folder_name in MN10)):
                continue
            files = glob(osp.join(folder, "test", "*.txt"))
            for file in files:
                data = self.read_func(file, graph_type2)
                datas2.append(data)
            graph_type2 += 1
        self.test_dataset = datas2
        
        if(self.dataset_type == "10"):
            self.num_classes = 10
        

        torch.save(self.train_dataset, self.processed_paths[0])
        torch.save(self.test_dataset, self.processed_paths[1])


    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['train.pt', 'test.pt']

    def read_graph(self, path, graph_type):
        f = open(path)
        x = []
        _y = [graph_type]
        y = torch.tensor(_y, dtype=torch.long)
        edge_index = [[], []]
        edge_attr = []
        nodeNum = (int)(f.readline())
        for i in range(nodeNum):
            line = f.readline()
            if(self.need_r):
                x.append([float(x) for x in line.split()][:4])
            else:
                x.append([float(x) for x in line.split()][:3])
        
        

        linkNum = (int)(f.readline())
        links = []
        soft_link = [[] for i in range(len(x))]
        for i in range(linkNum):
            line = f.readline()
            type, _x, _y, _len = line.split()
            links.append([_x, _y, _len])
            soft_link[(int)(_x)].append([(int)(_y), (float)(_len)])
            soft_link[(int)(_y)].append([(int)(_x), (float)(_len)])
            edge_index[0].append((int)(_x))
            edge_index[1].append((int)(_y))
            edge_index[0].append((int)(_y))
            edge_index[1].append((int)(_x))
            edge_attr.append([float(_len)])
            edge_attr.append([float(_len)])
        
        if(self.explict_feature):
            sorter = lambda x:(x[1],x[0])
            pos_ = []
            abs_ = []
            dis_ = []
            for i in range(len(x)):
                soft_link[i].sort(key = sorter, reverse = True)
                j=0
                app_pos = []
                app_abs = []
                app_dis = []
                for k in soft_link[i]:
                    app_pos.extend(x[k[0]])
                    app_abs.extend([x[i][0]-x[k[0]][0], x[i][1]-x[k[0]][1], x[i][2]-x[k[0]][2], x[i][3]-x[k[0]][3]])
                    app_dis.append(k[1])
                    j+=1
                    if(j ==6):
                        break
                if(j < 6):
                    app_pos.extend([0]*((6-j)*4))
                    app_abs.extend([0]*((6-j)*4))
                    app_dis.extend([0]*(6-j))
                pos_.append(app_pos)
                abs_.append(app_abs)
                dis_.append(app_dis)
                pass
            for i in range(len(x)):
                x[i].extend(pos_[i])
            for i in range(len(x)):
                x[i].extend(abs_[i])
            for i in range(len(x)):
                x[i].extend(dis_[i])
        
        if(self.to_1_regularize == True):
            x = np.array(x)
            x_r = x[:,3]
            x = x[:,0:3]
            xM = x.max(axis=0)
            xm = x.min(axis=0)
            x_m = np.where(-xm > xM, -xm, xM).max()
            if(x_m == 0):
                x_m = 1
            x /= x_m
            x_r = np.array([np.tanh(32*x_r)]).T
            x = np.concatenate((x, x_r), axis=1)
        else:
            x = np.array(x)
            if(self.need_r == True):
                if(self.explict_feature == False):
                    x[:,3] = np.tanh(4*x[:,3])
                else:
                    x[:, (3,7,11,15,19,23,27,31,35,39,43,47,51)] *= 8

        x = torch.tensor(x, dtype = torch.float)
        edge_index = torch.tensor(edge_index, dtype = torch.long)
        edge_attr = torch.tensor(edge_attr, dtype = torch.float)
        if(self.type == "gcn"):
            data = Data(x = x, edge_index = edge_index, edge_attr = edge_attr, y = y)
        return data

    def read_graph_norotate(self, path, graph_type):
        f = open(path)
        x = []
        _y = [graph_type]
        y = torch.tensor(_y, dtype=torch.long)
        edge_index = [[], []]
        nodeNum = (int)(f.readline())
        for i in range(nodeNum):
            line = f.readline()
            _x = [float(x) for x in line.split()]
            x.append(_x)
        linkNum = (int)(f.readline())
        links = []
        soft_link = [[] for i in range(len(x))]
        for i in range(linkNum):
            line = f.readline()
            _x, _y, _len = line.split()
            links.append([_x, _y, _len])
            edge_index[0].append((int)(_x))
            edge_index[1].append((int)(_y))
            edge_index[0].append((int)(_y))
            edge_index[1].append((int)(_x))
        if(self.to_1_regularize == True):
            x = np.array(x)
            x[np.isnan(x)] = 0
            xM = x.max(axis=0)
            xm = x.min(axis=0)
            x_m = np.where(-xm > xM, -xm, xM)
            x_m[x_m == 0] = 1
            x /= x_m
        x = torch.tensor(x, dtype = torch.float)
        edge_index = torch.tensor(edge_index, dtype = torch.long)
        data = Data(x = x, edge_index = edge_index, y = y)
        return data