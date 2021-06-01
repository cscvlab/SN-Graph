import os
import os.path as osp
import random
from tqdm import tqdm
import numpy as np
import time
import pathlib

import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from ModelNet40sdfsphPre import ModelNetSdfSpherePre
from ModelNet40sdfsphRotate import ModelNetSdfSphereRotate
from torch_geometric.data import DataLoader
import gatmodel
import networks

############################
path256 = ""
############################
path = path256
num_classes = 40
node_per_graph = 256
graph_batch_size = 64
no_rotate = False

input_feature = 4
needr = True
exf = False

regul_ = False

lrate = 0.001
############################

print(torch.cuda.is_available())
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = gatmodel.NetGAT(node_per_graph, input_feature, num_classes).to(device)
modelname = "GAT"

optimizer = torch.optim.Adam(model.parameters(), lr=lrate)
optimizerSGD = torch.optim.SGD(model.parameters(), lr=lrate, momentum=0.9, weight_decay = 1e-4)
scheduler = StepLR(optimizer, step_size = 20, gamma = 0.7)
schedulerSGD = StepLR(optimizerSGD, step_size = 20, gamma = 0.7)
optimSwitch = 250

#########################
epoch_num = 250
#########################

model_name = modelname
now_time = time.strftime("%Y%m%d-%H-%M", time.localtime())
pathlib.Path('./experiment/EXP_{}_{}_{}'.format(model_name, node_per_graph, now_time)).mkdir(parents=True, exist_ok=True)

dataset = ModelNetSdfSpherePre(path, type = "gcn", dataset_type="40", explict_feature=exf, to_1_regularize= regul_, 
                            need_r = needr, no_rotate=no_rotate)
# dataset = ModelNetSdfSphereRotate(path)

train_dataset = dataset.train_dataset
random.shuffle(train_dataset)
test_dataset = dataset.test_dataset
test_loader = DataLoader(test_dataset, batch_size=graph_batch_size, shuffle=False)
train_loader = DataLoader(train_dataset, batch_size=graph_batch_size, shuffle=True)


LOG_FOUT = open("./experiment/EXP_{}_{}_{}/log.txt".format(model_name, node_per_graph, now_time), 'w+')

def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}

def log_string(out_str):
    LOG_FOUT.write(str(out_str)+'\n')
    LOG_FOUT.flush()
    print(out_str)

def train(epoch):
    model.train()

    loss_all = 0
    for data in tqdm(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        clf = model.train()
        output = clf(data)
        loss = F.nll_loss(output, data.y)
        loss.backward()
        loss_all += data.num_graphs * loss.item()
        if(epoch > optimSwitch):
            optimizerSGD.step()
        else:
            optimizer.step()
    return loss_all / len(train_dataset)


def test(loader, label_eval = False):
    model.eval()

    correct = 0
    loss_all = 0
    for data in tqdm(loader):
        data = data.to(device)
        with torch.no_grad():
            clf = model.eval()
            output = clf(data)
            loss = F.nll_loss(output, data.y)
            loss_all += data.num_graphs * loss.item()
            pred = output.max(dim=1)[1]
            correct += pred.eq(data.y).sum().item()
    return correct / len(loader.dataset), loss_all / len(loader.dataset)

best_val_acc = 0

print("\n\n")
print(get_parameter_number(model))

for epoch in range(1, epoch_num+1):
    model.spEval = False
    loss = train(epoch)

    train_acc, _ = test(train_loader, True)
    test_acc, test_loss = test(test_loader, True)
    if(epoch > optimSwitch):
        schedulerSGD.step()
        for param_group in optimizerSGD.param_groups:
            print("learning rate:", param_group['lr'])
    else:
        scheduler.step()
        for param_group in optimizer.param_groups:
            print("learning rate:", param_group['lr'])
    if test_acc > best_val_acc:
        best_val_acc = test_acc
        if(best_val_acc >= 0.50):
            mode_save_name = "{}-{}-epoch{}-acc{:.3}.pt".format(model_name, now_time, epoch, best_val_acc)
            save_path = osp.join("./experiment/EXP_{}_{}_{}/".format(model_name, node_per_graph, now_time), mode_save_name)
            torch.save(model.state_dict(), save_path)
    print('Epoch: {:03d}, Loss: {:.5f}, Train Acc: {:.5f}, Test Loss: {:.5f}, Test Acc: {:.5f}'.
            format(epoch, loss, train_acc, test_loss, test_acc))
    log_string('Epoch: {:03d}, Train Loss: {:.7f}, Train acc: {:.7f}, '
          ', Test Loss: {:.5f}, Test Acc: {:.7f}, Best Acc: {:.7f}'.format(epoch, loss, train_acc, test_loss, test_acc, best_val_acc))

torch.save(model, osp.join("./experiment/EXP_{}_{}_{}/".format(model_name, node_per_graph, now_time), "EntireModel"))
