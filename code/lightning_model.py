# utils
import json
import numpy as np
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator
from sklearn.metrics import accuracy_score

# torch
import torch
import torch.nn.functional as F
from torch_scatter import scatter_add, scatter_max, scatter_mean
from torch_geometric.data import DataLoader

from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from ogb.graphproppred import PygGraphPropPredDataset, Evaluator

from model import PANConv, PANPooling, PANDropout
from Norm import Norm


class BaseNet(LightningModule):
    def __init__(self, batch_size=32, lr=1e-3, weight_decay=1e-3, num_workers=0, 
                 criterion=torch.nn.BCEWithLogitsLoss(pos_weight = torch.tensor([30])),  epochs=20):
        super().__init__()

        # loading params
        with open('parameters.json') as json_file:
            parameters = json.load(json_file)
        self.configuration = parameters

       
        self.save_hyperparameters(
            dict(
                batch_size = parameters["batch_size"],
                lr=parameters["learning_rate"],
                weight_decay=parameters["weight_decay"],
                num_workers=parameters["num_workers"],
                criterion=criterion,
                epochs=parameters["epochs"],
            )
        )
        
        self._train_data = None
        self._collate_fn = None
        self._train_loader = None

        self.batch_size = batch_size
        self.num_workers = num_workers
        
        
        self.lr = lr
        self.epochs=epochs
        
        self.weight_decay = weight_decay
        self.criterion = criterion

        self.evaluator = Evaluator(parameters["dataset_name"])


    def configure_optimizers(self):
        optimizer =  torch.optim.Adam(self.parameters(), lr= self.lr, betas= (0.9,0.999), 
                          weight_decay= self.weight_decay)
        
        return optimizer
        
    def train_dataloader(self):
        dataset = PygGraphPropPredDataset(name='ogbg-molhiv')
        split_idx = dataset.get_idx_split()
        train_data = dataset[split_idx["train"]]
        train_loader = DataLoader(train_data,
        batch_size=self.configuration["batch_size"], shuffle=True,
        num_workers = self.configuration["num_workers"])

        self._train_data = train_data
        self._train_loader = train_loader
        
        return train_loader
            
    def val_dataloader(self):
        dataset = PygGraphPropPredDataset(name='ogbg-molhiv')
        split_idx = dataset.get_idx_split()
        val_data = dataset[split_idx["valid"]]
        validation_loader = DataLoader(val_data,
        batch_size=self.configuration["batch_size"], shuffle=False,
        num_workers = self.configuration["num_workers"])


        self._validation_data = val_data
        self._validation_loader = validation_loader
        
        return validation_loader

    def test_dataloader(self):
        dataset = PygGraphPropPredDataset(name='ogbg-molhiv')
        split_idx = dataset.get_idx_split()
        test_data = dataset[split_idx["test"]]
        test_loader = DataLoader(test_data,
        batch_size=self.configuration["batch_size"], shuffle=False,
        num_workers = self.configuration["num_workers"])


        self._test_data = test_data
        self._test_loader = test_loader
        
        return test_loader

class LightningPAN(BaseNet):
    def __init__(self, num_node_features, num_classes, nhid, ratio, filter_size):
        super(LightningPAN, self).__init__()
        self.conv1 = PANConv(num_node_features, nhid, filter_size)
        self.norm1 = Norm('gn', nhid)
        self.pool1 = PANPooling(nhid, filter_size=filter_size)
        self.drop1 = PANDropout()

        self.conv2 = PANConv(nhid, nhid, filter_size=2)
        self.norm2 = Norm('gn', nhid)
        self.pool2 = PANPooling(nhid)
        self.drop2 = PANDropout()

        self.conv3 = PANConv(nhid, nhid, filter_size=2)
        self.norm3 = Norm('gn', nhid)
        self.pool3 = PANPooling(nhid)

        self.lin1 = torch.nn.Linear(nhid, nhid//2)
        self.lin2 = torch.nn.Linear(nhid//2, nhid//4)
        self.lin3 = torch.nn.Linear(nhid//4, 1)
        
    def forward(self, data):

        x, edge_index, batch = data.x, data.edge_index, data.batch
        perm_list = list()
        edge_mask_list = None

        x = self.conv1(x, edge_index)
        M = self.conv1.m
        x = self.norm1(x, batch)
        x, edge_index, _, batch, perm, score_perm = self.pool1(x, edge_index, batch=batch, M=M)
        perm_list.append(perm)
        edge_mask_list = self.drop1(edge_index, p=0.5)

        x = self.conv2(x, edge_index, edge_mask_list=edge_mask_list)
        M = self.conv2.m
        x = self.norm2(x, batch)
        x, edge_index, _, batch, perm, score_perm = self.pool2(x, edge_index, batch=batch, M=M)
        perm_list.append(perm)
        edge_mask_list = self.drop2(edge_index, p=0.5)

        x = self.conv3(x, edge_index, edge_mask_list=edge_mask_list)
        M = self.conv3.m
        x = self.norm3(x, batch)
        x, edge_index, _, batch, perm, score_perm = self.pool3(x, edge_index, batch=batch, M=M)
        perm_list.append(perm)

        mean = scatter_mean(x, batch, dim=0)
        x = mean

        x = self.lin1(x)
        x = self.lin2(x)
        x = self.lin3(x)

        return x, perm_list
    
    def training_step(self, batch, batch_idx):
        if batch.x.shape[0] == 1 or batch.batch[-1] == 0:
            print("JUST PASSED SOMETHING")
            pass
        else:
            pred, _ = self(batch)
            is_labeled = batch.y == batch.y
            loss = self.criterion(pred.to(torch.float32)[is_labeled], batch.y.to(torch.float32)[is_labeled])
            y_true = batch.y.view(pred.shape).detach().cpu().numpy()
            y_pred = pred.detach().cpu().numpy()
        self.log("pool1_weight_X", self.pool1.pan_pool_weight[0])
        self.log("pool1_weight_diagM", self.pool1.pan_pool_weight[1])
        self.log("pool2_weight_X", self.pool2.pan_pool_weight[0])
        self.log("pool2_weight_diagM", self.pool2.pan_pool_weight[1])
        self.log("pool3_weight_X", self.pool3.pan_pool_weight[0])
        self.log("pool3_weight_diagM", self.pool3.pan_pool_weight[1])
        return {"loss": loss, "y_true": y_true, "y_pred": y_pred}

    def training_epoch_end(self, outputs):
        training_loss = np.array([])
        y_true = np.array([])
        y_pred = np.array([])
        for results_dict in outputs:
            training_loss = np.append(training_loss, results_dict["loss"])
            y_true = np.append(y_true, results_dict['y_true'])
            y_pred = np.append(y_pred, results_dict['y_pred'])
        # acc = accuracy_score(y_true, y_pred)
        input_dict = {"y_true": y_true.reshape(-1, 1), "y_pred": y_pred.reshape(-1, 1)}
        self.log('rocauc_train', (self.evaluator.eval(input_dict))['rocauc'])
        self.log('train_loss', training_loss.sum().item())
    
    def validation_step(self, batch, batch_idx):
        if batch.x.shape[0] == 1:
            pass
        else:
            with torch.no_grad():
                pred, _ = self(batch)
                loss = self.criterion(pred, batch.y.float())

            y_true = batch.y.view(pred.shape).detach().cpu()
            y_pred = pred.detach().cpu()

        return {"loss": loss, "y_true": y_true, "y_pred": y_pred}
    
    def validation_epoch_end(self, outputs):
        validation_loss = np.array([])
        y_true = np.array([])
        y_pred = np.array([])
        for results_dict in outputs:
            validation_loss = np.append(validation_loss, results_dict["loss"])
            y_true = np.append(y_true, results_dict['y_true'])
            y_pred = np.append(y_pred, results_dict['y_pred'])
        input_dict = {"y_true": y_true.reshape(-1, 1), "y_pred": y_pred.reshape(-1, 1)}
        self.log('rocauc_eval', (self.evaluator.eval(input_dict))['rocauc'])
        self.log('validation_loss', validation_loss.sum().item())


    def test_step(self, batch, batch_idx):
        if batch.x.shape[0] == 1:
            pass
        else:
            pred, _ = self(batch)
            loss = self.criterion(pred, batch.y.float())

            y_true = batch.y.view(pred.shape).detach().cpu()
            y_pred = pred.detach().cpu()

        return {"loss": loss, "y_true": y_true, "y_pred": y_pred}

    def test_epoch_end(self, outputs):
        test_loss = np.array([])
        y_true = np.array([])
        y_pred = np.array([])
        for results_dict in outputs:
            test_loss = np.append(test_loss, results_dict['loss'])
            y_true = np.append(y_true, results_dict['y_true'])
            y_pred = np.append(y_pred, results_dict['y_pred'])
        input_dict = {"y_true": y_true.reshape(-1, 1), "y_pred": y_pred.reshape(-1, 1)}
        self.log('rocauc_test', (self.evaluator.eval(input_dict))['rocauc'])
        self.log('test_loss', test_loss.sum().item())