import os
import os.path as osp
import shutil
from glob import glob

from tqdm import tqdm

import torch
from torch_geometric.data import Dataset, Data, InMemoryDataset
import numpy as np
import math

class ModelNetSdfSphereRotate(InMemoryDataset):
    def __init__(self, root, transform = None, pre_transform = None):
        self.root = root
        self.read_func = self.read_graph

        super(ModelNetSdfSphereRotate, self).__init__(root, transform, pre_transform)

        self.train_dataset = torch.load(self.processed_paths[0])
        self.test_dataset   = torch.load(self.processed_paths[1])

        

    def process(self):
        files = []
        datas = []
        self.label_name = []
        graph_type = 0
        folders = [f.path for f in os.scandir(self.root) if f.is_dir()]
        for folder in folders:
            folder_name = folder.split("/")[-1]
            if(folder_name == "processed"):
                continue
            self.label_name.append(folder_name)
            files = glob(osp.join(folder, "train", "*.txt"))
            for file in tqdm(files):
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
            files = glob(osp.join(folder, "test", "*.txt"))
            for file in tqdm(files):
                data = self.read_func(file, graph_type2)
                datas2.append(data)
            graph_type2 += 1
        self.test_dataset = datas2

        torch.save(self.train_dataset, self.processed_paths[0])
        torch.save(self.test_dataset, self.processed_paths[1])


    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['train_rotate.pt', 'test_rotate.pt']



    def read_graph(self, path, graph_type):
        f = open(path)
        x = []
        y = torch.tensor([graph_type], dtype=torch.long)
        edge_index = [[], []]
        edge_attr = []
        nodeNum = (int)(f.readline())
        for i in range(nodeNum):
            line = f.readline()
            x.append([float(x) for x in line.split()])
        
        

        linkNum = (int)(f.readline())
        links = []
        for i in range(linkNum):
            line = f.readline()
            _x, _y, _len = line.split()
            links.append([_x, _y, _len])
            edge_index[0].append((int)(_x))
            edge_index[1].append((int)(_y))
            edge_index[0].append((int)(_y))
            edge_index[1].append((int)(_x))
            edge_attr.append([float(_len)])
            edge_attr.append([float(_len)])
        
        x = np.array(x)
        x[np.isnan(x)] = 0
        xM = x.max(axis=0)
        xm = x.min(axis=0)
        x_m = np.where(-xm>xM, -xm, xM)
        x_m[x_m == 0] = 1
        x_m[0] = 1
        x_m[1] = 1
        x /= x_m

        x = torch.tensor(x, dtype = torch.float)
        edge_index = torch.tensor(edge_index, dtype = torch.long)
        edge_attr = torch.tensor(edge_attr, dtype = torch.float)
        data = Data(x = x, edge_index = edge_index, edge_attr = edge_attr, y = y)
        return data