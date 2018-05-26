from torch.autograd import Variable
from lib.distributions import Normal
import numpy as np


def eval_model(data, model, train_config, visualize=False):
    """
    Function to evaluate the model on data.

    Args:
        data (DataLoader): a data loader that provides batches of sequence data
        model (LatentVariableModel): model to train
    """

    backward = True

    out_dict = {}
    n_examples = len(data)
    # n_steps = data_config['sequence_length']-1
    n_inf_iter = train_config['inference_iterations']
    assert n_inf_iter > 0, 'Number of inference iterations must be positive.'

    n_seq_samples = train_config['sequence_samples']
    n_step_samples = train_config['step_samples']

    # to record the statistics while running
    out_dict['free_energy']    = np.zeros((n_examples, n_inf_iter+1))
    out_dict['cond_log_like']  = np.zeros((n_examples, n_inf_iter+1))
    out_dict['kl_div']         = np.zeros((n_examples, n_inf_iter+1))
    out_dict['out_mean']       = []
    out_dict['out_log_var']    = []
    # out_dict['out_log_var']    = np.zeros((n_batches, n_inf_iter+1))
    # out_dict['mean_grad']      = np.zeros((n_batches, n_inf_iter+1))
    # out_dict['log_var_grad']   = np.zeros((n_batches, n_inf_iter+1))

    # loop over training examples
    for example_ind, example in enumerate(data):
        print('Example: ' + str(example_ind+1) + ' of ' + str(len(data)))

        # re-initialize the model from the data
        example = Variable(example.cuda())
        # example = Variable(example.cuda(), volatile=True)
        # batch = Variable(batch)
        model.re_init(example[0])

        n_steps = example.data.shape[0]
        # to record the statistics while running on this example
        step_free_energy     = np.zeros((n_inf_iter+1, n_steps))
        step_cond_log_like   = np.zeros((n_inf_iter+1, n_steps))
        step_kl_div          = np.zeros((n_inf_iter+1, n_steps))
        step_output_mean     = np.zeros((n_inf_iter+1, n_steps, example.data.shape[-1]))
        if type(model.output_dist) == Normal:
            pass
            # step_output_log_var  = np.zeros((n_inf_iter+1, n_steps, example.data.shape[-1]))
        # step_mean_grad       = np.zeros((n_inf_iter+1, n_steps))
        # step_log_var_grad    = np.zeros((n_inf_iter+1, n_steps))

        # loop over sequence steps
        for step_ind, step_example in enumerate(example[1:]):

            # set the mode to inference
            model.inference_mode()
            model._detach_h = True

            # generate a prediction
            model.generate(n_samples=n_step_samples)

            # evaluate the free energy to get gradients, errors
            free_energy, cond_log_like, kl = model.losses(step_example, averaged=False)
            if backward:
                free_energy.mean(dim=0).backward(retain_graph=True)

            step_free_energy[0, step_ind]    = free_energy.data.cpu().numpy()
            step_cond_log_like[0, step_ind]  = cond_log_like.data.cpu().numpy()
            step_kl_div[0, step_ind]         = kl[0].data.cpu().numpy()
            step_output_mean[0, step_ind]    = model.output_dist.mean.mean(dim=1).data.cpu().numpy()
            if type(model.output_dist) == Normal:
                pass
                # step_output_log_var[0, step_ind] = model.output_dist.log_var.mean(dim=1).data.cpu().numpy()
            # if type(model.output_dist) == Normal:
            #     if model.output_dist.log_var.data.shape == 3:
            #         step_output_log_var[:, 0, step_ind] = model.output_dist.log_var.mean(dim=2).mean(dim=1).data.cpu().numpy()
            # step_mean_grad[:, 0, step_ind]      = model.latent_levels[0].latent.approx_posterior_gradients()[0].abs().mean(dim=1).data.cpu().numpy()
            # step_log_var_grad[:, 0, step_ind]   = model.latent_levels[0].latent.approx_posterior_gradients()[1].abs().mean(dim=1).data.cpu().numpy()

            # iterative inference
            for inf_it in range(n_inf_iter):
                # perform inference
                model.infer(step_example)

                # generate a reconstruction
                model.generate(n_samples=n_step_samples)

                # evaluate the free energy to get gradients, errors
                free_energy, cond_log_like, kl = model.losses(step_example, averaged=False)
                if backward:
                    free_energy.mean(dim=0).backward(retain_graph=True)

                step_free_energy[inf_it+1, step_ind]    = free_energy.data.cpu().numpy()
                step_cond_log_like[inf_it+1, step_ind]  = cond_log_like.data.cpu().numpy()
                step_kl_div[inf_it+1, step_ind]         = kl[0].data.cpu().numpy()
                step_output_mean[inf_it+1, step_ind]    = model.output_dist.mean.mean(dim=1).data.cpu().numpy()
                if type(model.output_dist) == Normal:
                    pass
                    # step_output_log_var[inf_it+1, step_ind] = model.output_dist.log_var.mean(dim=1).data.cpu().numpy()
                # if type(model.output_dist) == Normal:
                #     if model.output_dist.log_var.data.shape == 3:
                #         step_output_log_var[:, inf_it+1, step_ind] = model.output_dist.log_var.mean(dim=2).mean(dim=1).data.cpu().numpy()
                # step_mean_grad[:, inf_it+1, step_ind]      = model.latent_levels[0].latent.approx_posterior_gradients()[0].abs().mean(dim=1).data.cpu().numpy()
                # step_log_var_grad[:, inf_it+1, step_ind]   = model.latent_levels[0].latent.approx_posterior_gradients()[1].abs().mean(dim=1).data.cpu().numpy()

                if np.isnan(step_free_energy[inf_it+1, step_ind].mean()):
                    # if nan is encountered, stop training
                    print('nan encountered during evaluation.')
                    import ipdb; ipdb.set_trace()

            # set the mode to generation
            model.generative_mode()
            model._detach_h = True

            # run the generative model
            model.generate(n_samples=n_step_samples)

            # form the prior on the next step
            model.step()

        out_dict['free_energy'][example_ind]   = step_free_energy.mean(axis=1)
        out_dict['cond_log_like'][example_ind] = step_cond_log_like.mean(axis=1)
        out_dict['kl_div'][example_ind]        = step_kl_div.mean(axis=1)
        out_dict['out_mean'].append(step_output_mean)
        if type(model.output_dist) == Normal:
            # out_dict['out_log_var'].append(step_output_log_var)
            pass
        # out_dict['out_log_var'][batch_ind]   = step_output_log_var.mean(axis=0)
        # out_dict['mean_grad'][batch_ind]     = step_mean_grad.mean(axis=2).mean(axis=0)
        # out_dict['log_var_grad'][batch_ind]  = step_log_var_grad.mean(axis=2).mean(axis=0)

    return out_dict
