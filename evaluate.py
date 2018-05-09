import torch
from config import run_config, train_config, data_config, model_config
from util.logging import Logger
from util.data.load_data import load_data
from lib.models import load_model

# hack to prevent the data loader from going on GPU 0
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=str(run_config['cuda_device'])
# torch.cuda.set_device(run_config['cuda_device'])
torch.cuda.set_device(0)

# TODO: write log likelihood evaluation script
