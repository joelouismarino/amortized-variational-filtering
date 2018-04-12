from torch.autograd import Variable
from config import run_config, data_config

import time
import numpy as np


def train(data, model, optimizers):
    """
    Function to train the model on data and update using optimizers.

    Args:
        data (DataLoader): a data loader that provides batches of sequence data
        model (LatentVariableModel): model to train
        optimizers (tuple): inference and generative optimizers respectively
    """
    # TODO: only step the optimizers at the end of the sequence
    # TODO: figure out smart way to store gradients during inference
    inf_opt, gen_opt = optimizers
    model.train()

    out_dict = {}
    n_batches = len(data)
    n_steps = data_config['sequence_length']
    n_inf_iter = run_config['inference_iterations']
    out_dict['free_energy']   = np.zeros((n_batches, n_steps))
    out_dict['cond_log_like'] = np.zeros((n_batches, n_steps))
    out_dict['kl_div']        = np.zeros((n_batches, n_steps))
    out_dict['out_log_var']   = np.zeros(n_batches)

    # loop over training examples
    for batch_ind, batch in enumerate(data):
        if batch.shape[1] == run_config['batch_size']:  ## hack for now
            print('Iteration: ' + str(batch_ind) + ' of ' + str(len(data)))
            # re-initialize the model from the data
            batch = Variable(batch.cuda())
            model.re_init(batch[0])

            # clear all of the gradients
            inf_opt.zero_stored_grad(); inf_opt.zero_current_grad()
            gen_opt.zero_stored_grad(); gen_opt.zero_current_grad()

            batch_size = batch.data.shape[1]
            step_free_energy   = np.zeros((batch_size, n_steps))
            step_cond_log_like = np.zeros((batch_size, n_steps))
            step_kl_div        = np.zeros((batch_size, n_steps))
            total_reconstruction = np.zeros(batch.data.shape)

            total_free_energy = 0.

            # loop over sequence steps
            for step_ind, step_batch in enumerate(batch[1:]):

                # set the mode to inference
                model.inference_mode()

                # form a prediction
                model.generate(gen=True)

                # get gradients/errors from free energy
                model.free_energy(step_batch).backward(retain_graph=True)

                # initial inference iteration
                model.infer(step_batch)

                # iterative inference
                for inf_it in range(n_inf_iter - 1):

                    model.generate()

                    model.free_energy(step_batch).backward(retain_graph=True)

                    model.infer(step_batch)

                inf_opt.collect()

                # set the mode to generation
                model.generative_mode()

                # evaluate the free energy, add to total
                total_free_energy += model.free_energy(step_batch)


                # free_energy_loss += free_energy.mean(dim=0)

                step_free_energy[:, step_ind]   = free_energy.data.cpu().numpy()
                step_cond_log_like[:, step_ind] = cond_log_like.data.cpu().numpy()
                step_kl_div[:, step_ind]        = kl[0].data.cpu().numpy()

                total_reconstruction[step_ind] = model.output_dist.mean.data.cpu().numpy()[:, 0]

                # form the prior on the next step
                model.step()

            if np.isnan(free_energy_loss.data.cpu().numpy()):
                print('Nan encountered during training...')
                import ipdb; ipdb.set_trace()

            free_energy_loss.backward()

            inf_opt.step()
            gen_opt.step()

            out_dict['free_energy'][batch_ind]   = step_free_energy.mean(axis=0)
            out_dict['cond_log_like'][batch_ind] = step_cond_log_like.mean(axis=0)
            out_dict['kl_div'][batch_ind]        = step_kl_div.mean(axis=0)
            out_dict['out_log_var'][batch_ind]   = model.output_dist.log_var.data.cpu().numpy().mean()

    out_dict['lr'] = (inf_opt.opt.param_groups[0]['lr'], gen_opt.opt.param_groups[0]['lr'])

    return out_dict


def validate(data, model):
    """
    Function to validate the model on data and update using optimizers and schedulers.

    Args:
        data (DataLoader): a data loader that provides batches of sequence data
        model (LatentVariableModel): model to train
    """

    model.eval()

    for batch_ind, batch in enumerate(data):
        model.re_init()
        for step_ind, step_batch in enumerate(batch):
            model.infer(Variable(step_batch))
            model.generate()
            model.step()
