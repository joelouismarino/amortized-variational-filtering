import torch
from config import run_config, train_config, data_config, model_config
from util.logging import Logger
from util.data.load_data import load_data
from lib.models import load_model
from util.eval import eval_model

# hack to prevent the data loader from going on GPU 0
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=str(run_config['cuda_device'])
# torch.cuda.set_device(run_config['cuda_device'])
torch.cuda.set_device(0)

logger = Logger(run_config)

# load the data
train_data, val_data, test_data = load_data(data_config, train_config['batch_size'])
if test_data is None:
    test_data = val_data

# load the model
print('Loading model...')
model = load_model(model_config)

assert run_config['resume_path'] is not None, 'Run path must be set for evaluation.'
print('Loading checkpoint ' + run_config['resume_path'])
model = logger.load_best(model)
# model = logger.load_epoch(model, 700)

print('Putting the model on the GPU...')
model.cuda()

model.eval()

output = eval_model(test_data, model, train_config)
