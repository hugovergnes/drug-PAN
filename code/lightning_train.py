import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from scikitplot.metrics import plot_confusion_matrix


from pytorch_lightning import Trainer
from lightning_model import LightningPAN
from pytorch_lightning.loggers.neptune import NeptuneLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint


# loading params
with open('parameters.json') as json_file:
    parameters = json.load(json_file)

datasetname = parameters["dataset_name"]
phi = parameters["phi"]
runs = parameters["runs"]
batch_size = parameters["batch_size"]
filter_size = parameters["maximum_path_size"]+1
learning_rate = parameters["learning_rate"]
weight_decay = parameters["weight_decay"]
pool_ratio = parameters["pool_ratio"]
topk_ratio = parameters['topk_ratio']
nhid = parameters["nhid"]
epochs = parameters["epochs"]
criterion_pos_weight = parameters['criterion_pos_weight']
LearningRateMonitor_Params = {'logging_interval': 'epoch'}

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Device: {}'.format(device))

model = LightningPAN(
    num_node_features=9, num_classes=1, nhid=nhid, 
    pool_ratio=pool_ratio, topk_ratio=topk_ratio, 
    criterion_pos_weight=criterion_pos_weight, filter_size=filter_size)

lr_logger = LearningRateMonitor(**LearningRateMonitor_Params)
neptune_logger = NeptuneLogger(
                api_key="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vdWkubmVwdHVuZS5haSIsImFwaV91cmwiOiJodHRwczovL3VpLm5lcHR1bmUuYWkiLCJhcGlfa2V5IjoiMDk5YjVmYzYtNTU0My00MzhkLWJiYTAtMGM4ZGVhZmEyMTZiIn0=",
                project_name='hvergnes/PAN',
                close_after_fit=False,
                params=parameters, # your hyperparameters, immutable
                tags=['PAN', 'change pool, topk, pos_weight, batch_size', 'fix epoch end'],  # tags
                upload_source_files=["parameters.json", "lightning_model.py"]
                )


trainer = Trainer(
    max_epochs=epochs,
    logger=neptune_logger,
    callbacks=[lr_logger],
    # fast_dev_run=True,
)

trainer.fit(model)
trainer.test(model)

test_loader = model.test_dataloader()
y_true = np.array([])
y_pred = np.array([])

for i, batch in enumerate(test_loader):
    y = batch.y.cpu().detach().numpy()
    y_hat, _ = model(batch)
    y_hat = y_hat.cpu().detach().numpy()
    y_hat = np.array([1 if value > 0.5 else 0 for value in y_hat])

    y_true = np.append(y_true, y)
    y_pred = np.append(y_pred, y_hat)

fig, ax = plt.subplots(figsize=(16, 12))
plot_confusion_matrix(y_true, y_pred, ax=ax)
neptune_logger.experiment.log_image('confusion_matrix', fig)