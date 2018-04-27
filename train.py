import torch
from config import run_config, train_config, data_config, model_config
from util.logging import Logger
from util.plotting import Plotter
from util.data.load_data import load_data
from lib.models import load_model
from util.optimization import load_opt_sched
from util.train_val import train, validate

# hack to prevent the data loader from going on GPU 0
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=str(run_config['cuda_device'])
# torch.cuda.set_device(run_config['cuda_device'])
torch.cuda.set_device(0)

# initialize logging and plotting
logger = Logger(run_config)
plotter = Plotter(logger.log_dir)

# load the data
train_data, val_data, test_data = load_data(data_config, train_config['batch_size'])

# load the model, optimizers
print('Loading model...')
model = load_model(model_config)
print('Loading optimizers...')
optimizers, schedulers = load_opt_sched(train_config, model)

if run_config['resume_path']:
    print('Resuming checkpoint ' + run_config['resume_path'])
    model, optimizers, schedulers = logger.load_checkpoint(model, optimizers)

print('Putting the model on the GPU...')
model.cuda()

while True:
    # training
    # out = train(train_data, model, optimizers)
    out = train(test_data, model, optimizers)
    logger.log(out, 'Train'); plotter.plot(out, 'Train')
    if val_data:
        # validation
        out = validate(val_data, model)
        logger.log(out, 'Val'); plotter.plot(out, 'Val')
    if logger.save_epoch():
        logger.save_checkpoint(model, optimizers)
    logger.step(); plotter.step()
    schedulers[0].step(); schedulers[1].step()
    plotter.save()
