import os
import torch
from time import strftime
from util.optimization import load_sched
from log_util import set_gpu_recursive, get_last_epoch, update_metric


class Logger(object):

    def __init__(self, run_config):
        """
        Initialize logs, either from existing experiment or create new experiment.
        """
        print('Initializing logs...')
        log_root = run_config['log_root_path']
        self._save_iter = run_config['save_iter']
        if run_config['resume_path']:
            # resume an old experiment
            self.log_dir = run_config['resume_path']
            if os.path.exists(os.path.join(log_root, self.log_dir)):
                self.log_path = os.path.join(log_root, self.log_dir)
                print(' Resuming experiment ' + self.log_dir)
            else:
                raise Exception('Experiment folder ' + self.log_dir + ' not found.')
        else:
            # start a new experiment
            self.log_dir = strftime("%b_%d_%Y_%H_%M_%S") + '/'
            self.log_path = os.path.join(log_root, self.log_dir)
            os.makedirs(self.log_path)
            os.system("rsync -au --include '*/' --include '*.py' --exclude '*' . " + self.log_path + "source")
            os.makedirs(os.path.join(self.log_path, 'metrics'))
            os.makedirs(os.path.join(self.log_path, 'checkpoints'))
            self.epoch = 1
            print(' Starting experiment ' + self.log_dir)

    def save_epoch(self):
        """
        Specifies whether or not to save the model at the current epoch.
        """
        # TODO: use the validation loss to override this?
        return (self.epoch % self._save_iter) == 0

    def save_checkpoint(self, model, optimizers):
        """
        Save the model, optimizers.

        Args:
            model (LatentVariableModel): model to save
            optimizers (tuple): inference and generative optimizers
        """
        ckpt_path = os.path.join(self.log_path, 'checkpoints', str(self.epoch))
        if not os.path.exists(ckpt_path):
            os.makedirs(ckpt_path)
        # TODO: put everything on CPU first
        torch.save(model.state_dict(), os.path.join(ckpt_path, 'model.ckpt'))
        torch.save(tuple([optimizer.opt.state_dict() for optimizer in optimizers]),
                    os.path.join(ckpt_path, 'opt.ckpt'))

    def load_checkpoint(self, model, optimizers):
        """
        Load the model and optimizers from the most recent epoch.

        Args:
            model (LatentVariableModel): model to load
            optimizers (tuple): inference and generative optimizers
        """
        self.epoch = get_last_epoch(self.log_path)

        model_state_dict = torch.load(os.path.join(self.log_path, 'checkpoints', str(self.epoch), 'model.ckpt'))
        model.load_state_dict(model_state_dict)

        optimizer_state_dict = torch.load(os.path.join(self.log_path, 'checkpoints', str(self.epoch), 'opt.ckpt'))
        for opt_ind in range(len(optimizers)):
            optimizers[opt_ind].opt.load_state_dict(optimizer_state_dict[opt_ind])
            optimizers[opt_ind].opt.state = set_gpu_recursive(optimizers[opt_ind].state, torch.cuda.current_device())

        schedulers = load_sched(optimizers, self.epoch)

        return model, optimizers, schedulers

    def log(self, out_dict, train_val):
        """
        Function to log results.
        """
        train_val = train_val.lower()
        update_metric(os.path.join(self.log_path, 'metrics', train_val + '_free_energy.p'),
                      (self.epoch, out_dict['free_energy']))
        update_metric(os.path.join(self.log_path, 'metrics', train_val + '_cond_log_like.p'),
                      (self.epoch, out_dict['cond_log_like']))
        update_metric(os.path.join(self.log_path, 'metrics', train_val + '_kl_div.p'),
                      (self.epoch, out_dict['kl_div']))

    def step(self):
        self.epoch += 1
