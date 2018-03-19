import os
import torch
from misc import set_gpu_recursive, get_last_epoch

global log_path


def init_log(run_config):
    """
    Initialize logs, either from existing experiment or create new experiment.
    """
    global log_path
    if run_config['resume_path']:
        # resume an old experiment
        log_root = run_config['experiment_path']
        if os.path.exists(os.path.join(log_root, run_config['resume_path'])):
            log_path = os.path.join(log_root, run_config['resume_path'])
            return run_config['resume_path']
        else:
            raise Exception('Experiment folder ' + run_config['resume_path'] + ' not found.')
    else:
        # start a new experiment
        log_dir = strftime("%b_%d_%Y_%H_%M_%S") + '/'
        log_path = os.path.join(log_root, log_dir)
        os.makedirs(log_path)
        os.system("rsync -au --include '*/' --include '*.py' --exclude '*' . " + log_path + "source")
        os.makedirs(os.path.join(log_path, 'metrics'))
        os.makedirs(os.path.join(log_path, 'checkpoints'))
        return log_dir


def load_checkpoint(path, run_config):
    """
    Load the model, optimizers, and schedulers from the most recent epoch of path.
    """
    epoch = get_last_epoch(path)
    model = torch.load(os.path.join(path, 'checkpoints', str(epoch), 'model.ckpt'))
    optimizers = torch.load(os.path.join(path, 'checkpoints', str(epoch), 'opt.ckpt'))
    for opt in optimizers:
        opt.state = set_gpu_recursive(opt.state, run_config['cuda_device'])
    schedulers = torch.load(os.path.join(path, 'checkpoints', str(epoch), 'sched.ckpt'))
    return model, optimizers, schedulers


def save_checkpoint(model, optimizers, schedulers):
    """
    Save the model, optimizers, and schedulers.
    """
    global log_path
    epoch = 0
    ckpt_path = os.path.join(log_path, 'checkpoints', str(epoch))
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)
    # TODO: put everything on CPU first
    torch.save(model, os.path.join(ckpt_path, 'model.ckpt'))
    torch.save(tuple(optimizers), os.path.join(ckpt_path, 'opt.ckpt'))
    torch.save(tuple(schedulers), os.path.join(ckpt_path, 'sched.ckpt'))
