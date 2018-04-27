from torch.autograd import Variable
from config import run_config, train_config, data_config
import numpy as np

global epoch
epoch = 0


def train(data, model, optimizers):
    """
    Function to train the model on data.
    """
    return run(data, model, optimizers)


def validate(data, model):
    """
    Function to validate the model on data.
    """
    return run(data, model)


def run(data, model, optimizers=None):
    """
    Function to train/validate the model on data.

    Args:
        data (DataLoader): a data loader that provides batches of sequence data
        model (LatentVariableModel): model to train
        optimizers (optional, tuple): inference and generative optimizers respectively
    """
    if optimizers:
        global epoch
        inf_opt, gen_opt = optimizers
        model.train()
        if train_config['kl_annealing_epochs'] > 0:
            if epoch < train_config['kl_annealing_epochs']:
                anneal_weight = epoch * 1. / train_config['kl_annealing_epochs']
            else:
                anneal_weight = 1.
        else:
            anneal_weight = 1.
        epoch += 1
    else:
        model.eval()
        anneal_weight = 1.

    out_dict = {}
    n_batches = len(data)
    n_steps = data_config['sequence_length']-1
    n_inf_iter = run_config['inference_iterations']
    assert n_inf_iter > 0, 'Number of inference iterations must be positive.'
    # to record the statistics while running
    out_dict['free_energy']    = np.zeros((n_batches, n_inf_iter+1, n_steps))
    out_dict['cond_log_like']  = np.zeros((n_batches, n_inf_iter+1, n_steps))
    out_dict['kl_div']         = np.zeros((n_batches, n_inf_iter+1, n_steps))
    out_dict['out_log_var']    = np.zeros((n_batches, n_inf_iter+1, n_steps))
    out_dict['mean_grad']      = np.zeros((n_batches, n_inf_iter+1))
    out_dict['log_var_grad']   = np.zeros((n_batches, n_inf_iter+1))
    if optimizers:
        out_dict['inf_param_grad'] = np.zeros(n_batches)
        out_dict['gen_param_grad'] = np.zeros(n_batches)

    # loop over training examples
    for batch_ind, batch in enumerate(data):
        print('Iteration: ' + str(batch_ind+1) + ' of ' + str(len(data)))
        # re-initialize the model from the data
        batch = Variable(batch.cuda())
        model.re_init(batch[0])

        # clear all of the gradients
        if optimizers:
            inf_opt.zero_stored_grad(); inf_opt.zero_current_grad()
            gen_opt.zero_stored_grad(); gen_opt.zero_current_grad()

        batch_size = batch.data.shape[1]
        # to record the statistics while running on this batch
        step_free_energy     = np.zeros((batch_size, n_inf_iter+1, n_steps))
        step_cond_log_like   = np.zeros((batch_size, n_inf_iter+1, n_steps))
        step_kl_div          = np.zeros((batch_size, n_inf_iter+1, n_steps))
        step_output_log_var  = np.zeros((batch_size, n_inf_iter+1, n_steps))
        step_mean_grad       = np.zeros((batch_size, n_inf_iter+1, n_steps))
        step_log_var_grad    = np.zeros((batch_size, n_inf_iter+1, n_steps))
        total_reconstruction = np.zeros(batch.data.shape)

        # the total free energy for the batch of sequences
        total_free_energy = 0.

        # loop over sequence steps
        for step_ind, step_batch in enumerate(batch[1:]):

            # set the mode to inference
            model.inference_mode()

            # clear the inference model's current gradients
            if optimizers:
                inf_opt.zero_current_grad()

            # generate a prediction
            model.generate()

            # evaluate the free energy to get gradients, errors
            free_energy, cond_log_like, kl = model.losses(step_batch, averaged=False, anneal_weight=anneal_weight)
            free_energy.mean(dim=0).backward(retain_graph=True)

            step_free_energy[:, 0, step_ind]    = free_energy.data.cpu().numpy()
            step_cond_log_like[:, 0, step_ind]  = cond_log_like.data.cpu().numpy()
            step_kl_div[:, 0, step_ind]         = kl[0].data.cpu().numpy()
            step_output_log_var[:, 0, step_ind] = model.output_dist.log_var.mean(dim=2).mean(dim=1).data.cpu().numpy()
            step_mean_grad[:, 0, step_ind]      = model.latent_levels[0].latent.approx_posterior_gradients()[0].abs().mean(dim=1).data.cpu().numpy()
            step_log_var_grad[:, 0, step_ind]   = model.latent_levels[0].latent.approx_posterior_gradients()[1].abs().mean(dim=1).data.cpu().numpy()

            # iterative inference
            for inf_it in range(n_inf_iter):
                # perform inference
                model.infer(step_batch)

                # generate a reconstruction
                model.generate()

                # evaluate the free energy to get gradients, errors
                free_energy, cond_log_like, kl = model.losses(step_batch, averaged=False, anneal_weight=anneal_weight)
                free_energy.mean(dim=0).backward(retain_graph=True)

                step_free_energy[:, inf_it+1, step_ind]    = free_energy.data.cpu().numpy()
                step_cond_log_like[:, inf_it+1, step_ind]  = cond_log_like.data.cpu().numpy()
                step_kl_div[:, inf_it+1, step_ind]         = kl[0].data.cpu().numpy()
                step_output_log_var[:, inf_it+1, step_ind] = model.output_dist.log_var.mean(dim=2).mean(dim=1).data.cpu().numpy()
                step_mean_grad[:, inf_it+1, step_ind]      = model.latent_levels[0].latent.approx_posterior_gradients()[0].abs().mean(dim=1).data.cpu().numpy()
                step_log_var_grad[:, inf_it+1, step_ind]   = model.latent_levels[0].latent.approx_posterior_gradients()[1].abs().mean(dim=1).data.cpu().numpy()

                if np.isnan(step_free_energy[:, inf_it+1, step_ind].mean()):
                    # if nan is encountered, stop training
                    print('nan encountered during training.')
                    import ipdb; ipdb.set_trace()

            if optimizers:
                # collect the inference model gradients into the stored gradients
                inf_opt.collect()
                # increment the iterators the appropriate number of steps
                inf_opt.step_iter(n_inf_iter)
                gen_opt.step_iter()

                if True:
                    inf_opt.step()

            # set the mode to generation
            model.generative_mode()

            # run the generative model
            model.generate()

            # evaluate the free energy, add to total
            total_free_energy += model.free_energy(step_batch, anneal_weight=anneal_weight)

            total_reconstruction[step_ind] = model.output_dist.mean.data.cpu().numpy()[:, 0]

            # form the prior on the next step
            model.step()

        if np.isnan(total_free_energy.data.cpu().numpy()):
            # if nan is encountered, stop training
            print('nan encountered during training.')
            import ipdb; ipdb.set_trace()

        # clear the generative model's current gradients
        if optimizers:
            gen_opt.zero_current_grad()

        # get the gradients (for the generative model)
        total_free_energy.backward()

        if optimizers:
            # collect the gradients into the stored gradients
            gen_opt.collect()

            out_dict['inf_param_grad'][batch_ind] = np.mean([grad.abs().mean().data.cpu().numpy() for grad in inf_opt.stored_grads])
            out_dict['gen_param_grad'][batch_ind] = np.mean([grad.abs().mean().data.cpu().numpy() for grad in gen_opt.stored_grads])

            print(inf_opt._n_iter)
            print(gen_opt._n_iter)

            # apply the gradients to the inference and generative models
            # inf_opt.step()
            gen_opt.step()

        out_dict['free_energy'][batch_ind]   = step_free_energy.mean(axis=0)
        out_dict['cond_log_like'][batch_ind] = step_cond_log_like.mean(axis=0)
        out_dict['kl_div'][batch_ind]        = step_kl_div.mean(axis=0)
        out_dict['out_log_var'][batch_ind]   = step_output_log_var.mean(axis=0)
        out_dict['mean_grad'][batch_ind]     = step_mean_grad.mean(axis=2).mean(axis=0)
        out_dict['log_var_grad'][batch_ind]  = step_log_var_grad.mean(axis=2).mean(axis=0)

    # average over the batch dimension
    for item_key in out_dict:
        out_dict[item_key] = out_dict[item_key].mean(axis=0)

    # record the learning rate
    if optimizers:
        out_dict['lr'] = (inf_opt.opt.param_groups[0]['lr'], gen_opt.opt.param_groups[0]['lr'])

    return out_dict
