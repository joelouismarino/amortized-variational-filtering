import os, sys
import torch
import argparse
from config import run_config, train_config, data_config, model_config
from util.logging import Logger
from util.plotting import Plotter
from util.data.load_data import load_data
from lib.models import load_model
from util.optimization import load_opt_sched
from util.train_val import train, validate


def start_training(run_config, train_config, data_config, model_config):
    # hack to prevent the data loader from going on GPU 0
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=str(run_config['cuda_device'])
    # torch.cuda.set_device(run_config['cuda_device'])
    torch.cuda.set_device(0)

    # initialize logging and plotting
    logger = Logger(run_config)
    plotter = Plotter(logger.log_dir, run_config, train_config, model_config, data_config)

    # load the data
    train_data, val_data, test_data = load_data(data_config, train_config['batch_size'])
    if val_data is None:
        val_data = test_data

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
        out = train(train_data, model, optimizers, train_config, data_config)
        logger.log(out, 'Train'); plotter.plot(out, 'Train')
        if val_data:
            # validation
            out = validate(val_data, model, train_config, data_config)
            logger.log(out, 'Val'); plotter.plot(out, 'Val')
        if logger.save_epoch():
            logger.save_checkpoint(model, optimizers)
        logger.step(); plotter.step()
        schedulers[0].step(); schedulers[1].step()
        plotter.save()


if __name__=='__main__':

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument
    arg_parser.add_argument('--dataset', default=None, help='data set to train on')
    arg_parser.add_argument('--model', default=None, help='model type')
    arg_parser.add_argument('--inference', default=None, help='inference type, baseline or avf')
    args = arg_parser.parse_args()

    if args.dataset and args.model and args.inference:
        print('Loading pre-defined configuration files...')
        path_to_config = os.path.join(os.getcwd(), 'experiments', args.model, args.dataset, args.inference)
        sys.path.insert(0, path_to_config)

        # completely overwrite train_config and model_config
        from train_config import train_config
        from model_config import model_config

        # overwrite some fields in data_config
        from data_config import data_config as dc
        from data_config import clean_data_config

        data_config = {'data_path': data_config['data_path']}
        data_config['dataset_name'] = dc['dataset_name']
        data_config['data_type'] = dc['data_type']
        data_config['sequence_length'] = dc['sequence_length']
        clean_data_config(data_config)

    start_training(run_config, train_config, data_config, model_config)
