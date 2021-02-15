# utils
import json
import numpy as np
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator
import os

# torch
import torch
import torch.nn.functional as F
from torch.utils.data import random_split

# torch geometric
from torch_geometric.data import DataLoader, Data

# model
from model import PAN

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Device: {}'.format(device))

# loading params
with open('parameters.json') as json_file:
    parameters = json.load(json_file)

def train(model, train_loader, device):
    model.train()

    loss_all = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        output, _ = model(data)
        loss = F.nll_loss(output, data.y)
        loss.backward()
        loss_all += data.num_graphs * loss.item()
        optimizer.step()
        for name, param in model.named_parameters():
            # if 'pan_pool_weight' in name:
            #     param.data = param.data.clamp(0, 1)
            if 'panconv_filter_weight' in name:
                param.data = param.data.clamp(0, 1)
            if 'panpool_filter_weight' in name:
                param.data = param.data.clamp(0, 1)
    return loss_all / len(train_loader.dataset)


def test(model, loader, device):
    model.eval()

    correct = 0
    loss = 0.0
    for data in loader:
        data = data.to(device)
        out, _ = model(data)
        pred = out.max(dim=1)[1]
        correct += pred.eq(data.y).sum().item()
        label = data.y.squeeze(1)
        loss += F.nll_loss(out, label).item()*data.num_graphs
    return correct / len(loader.dataset), loss/len(loader.dataset)


datasetname = parameters["dataset_name"]
phi = parameters["phi"]
runs = parameters["runs"]
batch_size = parameters["batch_size"]
filter_size = parameters["maximum_path_size"]+1
learning_rate = parameters["learning_rate"]
weight_decay = parameters["weight_decay"]
pool_ratio = parameters["pool_ratio"]
nhid = parameters["nhid"]
epochs = parameters["epochs"]

train_loss = np.zeros((runs,epochs),dtype=np.float)
val_loss = np.zeros((runs,epochs),dtype=np.float)
val_acc = np.zeros((runs,epochs),dtype=np.float)
test_acc = np.zeros(runs,dtype=np.float)
min_loss = 1e10*np.ones(runs)

# dataset
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
sv_dir = 'dataset/save'
if not os.path.exists(sv_dir):
    os.makedirs(sv_dir)
path = os.path.join(os.path.abspath(''), 'dataset/', datasetname)

# Load the dataset 
dataset = PygGraphPropPredDataset(name='ogbg-molhiv')
split_idx = dataset.get_idx_split()
# Check task type
print('Task type: {}'.format(dataset.task_type))


print(len(dataset))
print(dataset.num_classes)
print(dataset.num_node_features)

num_classes = dataset.num_classes
num_node_features = dataset.num_node_features
num_edge = 0
num_node = 0
num_graph = len(dataset)

dataset1 = list()
for i in range(len(dataset)):
    data1 = Data(x=dataset[i].x, edge_index=dataset[i].edge_index, y=dataset[i].y)
    data1.num_node = dataset[i].num_nodes
    data1.num_edge = dataset[i].edge_index.size(1)
    num_node = num_node + data1.num_node
    num_edge = num_edge + data1.num_edge
    dataset1.append(data1)
dataset = dataset1

num_edge = num_edge*1.0/num_graph
num_node = num_node*1.0/num_graph

# generate training, validation and test data sets
num_training = int(num_graph*0.8)
num_val = int(num_graph*0.1)
num_test = num_graph - (num_training+num_val)

## train model
for run in range(runs):

    training_set, val_set, test_set = random_split(dataset, [num_training,num_val,num_test])
    train_loader = DataLoader(training_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    print('***** PAN for {}, phi {} *****'.format(datasetname,phi))
    print('#training data: {}, #test data: {}'.format(num_training,num_test))
    print('Mean #nodes: {:.1f}, mean #edges: {:.1f}'.format(num_node,num_edge))
    print('Network architectur: PC-PA')
    print('filter_size: {:d}, pool_ratio: {:.2f}, learning rate: {:.2e}, weight decay: {:.2e}, nhid: {:d}'.format(filter_size,pool_ratio,learning_rate,weight_decay,nhid))
    print('batchsize: {:d}, epochs: {:d}, runs: {:d}'.format(batch_size,epochs,runs))
    print('Device: {}'.format(device))

    ## train model
    model = PAN(num_node_features,num_classes,nhid=nhid,ratio=pool_ratio,filter_size=filter_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    for epoch in range(epochs):
        # training
        model.train()
        loss_all = 0
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            output, _ = model(data)
            label = data.y.squeeze(1)
            loss = F.nll_loss(output, label)
            loss.backward()
            loss_all += data.num_graphs * loss.item()
            optimizer.step()
            for name, param in model.named_parameters():
                # if 'pan_pool_weight' in name:
                #     param.data = param.data.clamp(0, 1)
                if 'panconv_filter_weight' in name:
                    param.data = param.data.clamp(0, 1)
                if 'panpool_filter_weight' in name:
                    param.data = param.data.clamp(0, 1)
        loss = loss_all / len(train_loader.dataset)   
        train_loss[run,epoch] = loss
        # validation
        val_acc_1, val_loss_1 = test(model,val_loader,device)
        val_loss[run,epoch] = val_loss_1
        val_acc[run,epoch] = val_acc_1
        print('Run: {:02d}, Epoch: {:03d}, Val loss: {:.4f}, Val acc: {:.4f}'.format(run+1,epoch+1,val_loss[run,epoch],val_acc[run,epoch]))
        if val_loss_1 < min_loss[run]:
            # save the model and reuse later in test
            torch.save(model.state_dict(), 'latest.pth')
            min_loss[run] = val_loss_1

    # test
    model.load_state_dict(torch.load('latest.pth'))
    test_acc[run], _ = test(model,test_loader,device)
    print('==Test Acc: {:.4f}'.format(test_acc[run]))

print('==Mean Test Acc: {:.4f}'.format(np.mean(test_acc)))

t1 = time.time()
sv = datasetname + '_pcpa_runs' + str(runs) + '_phi' + str(phi) + '_time' + str(t1) + '.mat'
sio.savemat(sv,mdict={'test_acc':test_acc,'val_loss':val_loss,'val_acc':val_acc,'train_loss':train_loss,'filter_size':filter_size,'learning_rate':learning_rate,'weight_decay':weight_decay,'nhid':nhid,'batch_size':batch_size,'epochs':epochs})

