# utils
import json
import numpy as np
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator
import os
from tqdm import tqdm
# from torch.utils.tensorboard import SummaryWriter
# writer = SummaryWriter()

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

cls_criterion = torch.nn.BCEWithLogitsLoss()

def eval(model, loader, device, evaluator):
    model.eval()

    y_true = []
    y_pred = []

    evaluation_loss = 0

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)

        if batch.x.shape[0] == 1:
            pass
        else:
            with torch.no_grad():
                pred, _ = model(batch)
                # label = batch.y.squeeze(1)
                # loss = F.nll_loss(pred, label).item()
                loss = cls_criterion(pred, batch.y.float())
                evaluation_loss += data.num_graphs * loss.item()



            pred = pred.max(dim=1)[1]
            y_true.append(batch.y.view(pred.shape).detach().cpu())
            y_pred.append(pred.detach().cpu())

    y_true = torch.cat(y_true, dim = 0).numpy().reshape(-1, 1)
    y_pred = torch.cat(y_pred, dim = 0).numpy().reshape(-1, 1)

    input_dict = {"y_true": y_true, "y_pred": y_pred}

    print(f"Number of predicted positive: {y_pred.sum()}")
    print(f"Number of real positive: {y_true.sum()}")

    return evaluator.eval(input_dict), evaluation_loss


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

# train_loss = np.zeros((runs,epochs),dtype=np.float)
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

# Need to populate num_node and num_edges !!
# dataset1 = list()
# for i in range(len(dataset)):
#     data1 = Data(x=dataset[i].x, edge_index=dataset[i].edge_index, y=dataset[i].y)
#     data1.num_node = dataset[i].num_nodes
#     data1.num_edge = dataset[i].edge_index.size(1)
#     num_node = num_node + data1.num_node
#     num_edge = num_edge + data1.num_edge
#     dataset1.append(data1)
# dataset = dataset1

num_edge = num_edge*1.0/num_graph
num_node = num_node*1.0/num_graph

# cls_criterion = torch.nn.BCEWithLogitsLoss()
## train model
for run in range(runs):
    evaluator = Evaluator(parameters["dataset_name"])

    train_loader = DataLoader(dataset[split_idx["train"]],
     batch_size=parameters["batch_size"], shuffle=True,
      num_workers = parameters["num_workers"])

    val_loader = DataLoader(dataset[split_idx["valid"]],
     batch_size=parameters["batch_size"],
     shuffle=False, num_workers = parameters["num_workers"])

    test_loader = DataLoader(dataset[split_idx["test"]], 
    batch_size=parameters["batch_size"], shuffle=False, 
    num_workers = parameters["num_workers"])

   
    print('***** PAN for {}, phi {} *****'.format(datasetname,phi))
    print('Mean #nodes: {:.1f}, mean #edges: {:.1f}'.format(num_node,num_edge))
    print('Network architectur: PC-PA')
    print('filter_size: {:d}, pool_ratio: {:.2f}, learning rate: {:.2e}, weight decay: {:.2e}, nhid: {:d}'.format(filter_size,pool_ratio,learning_rate,weight_decay,nhid))
    print('batchsize: {:d}, epochs: {:d}, runs: {:d}'.format(batch_size,epochs,runs))
    print('Device: {}'.format(device))

    ## train model
    model = PAN(num_node_features,num_classes,nhid=nhid,ratio=pool_ratio,filter_size=filter_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    #uses scheduler for training only
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=parameters["epochs"], 
    eta_min=10e-7)

    for epoch in range(epochs):
        # training
        print(f"train/ Epoch:{epoch}")
        model.train()
        train_loss = 0
        for data in tqdm(train_loader):
            data = data.to(device)
            optimizer.zero_grad()
            output, _ = model(data)
            # label = data.y.squeeze(1)
            # loss = F.nll_loss(output, label)
            loss = cls_criterion(output, data.y.float())
            loss.backward()
            train_loss += data.num_graphs * loss.item()
            optimizer.step()
            scheduler.step()
            for name, param in model.named_parameters():
                # if 'pan_pool_weight' in name:
                #     param.data = param.data.clamp(0, 1)
                if 'panconv_filter_weight' in name:
                    param.data = param.data.clamp(0, 1)
                if 'panpool_filter_weight' in name:
                    param.data = param.data.clamp(0, 1)
        loss = train_loss / len(train_loader.dataset)   
        # train_loss[run, epoch] = loss
        train_perf, _ = eval(model, train_loader, device, evaluator)
        print(f"Epoch: {epoch} Train Loss: {loss}")
        # validation
        validation_perf, validation_loss = eval(model, val_loader, device, evaluator)
        print(f"Epoch: {epoch} Validation Loss: {validation_loss}")
        # Test
        test_perf, test_loss = eval(model, test_loader, device, evaluator)
        print(f"Epoch: {epoch} Test Loss: {test_loss}")
        print(f"train_perf: {train_perf}")
        print(f"val_perf: {validation_perf}")
        print(f"val_perf: {test_perf}")
        if test_loss < min_loss[run]:
            # save the model dand reuse later in test
            torch.save(model.state_dict(), 'latest.pth')
            # min_loss[run] = validation_perf

    # test
    model.load_state_dict(torch.load('latest.pth'))
    test_perf = eval(model, test_loader, device, evaluator)
    print(f'==Test perf: {test_perf}')

# print('==Mean Test Acc: {:.4f}'.format(np.mean(test_acc)))

# t1 = time.time()
# sv = datasetname + '_pcpa_runs' + str(runs) + '_phi' + str(phi) + '_time' + str(t1) + '.mat'
# sio.savemat(sv,mdict={'test_acc':test_acc,'val_loss':val_loss,'val_acc':val_acc,'train_loss':train_loss,'filter_size':filter_size,'learning_rate':learning_rate,'weight_decay':weight_decay,'nhid':nhid,'batch_size':batch_size,'epochs':epochs})

