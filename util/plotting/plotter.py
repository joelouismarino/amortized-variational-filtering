import visdom
import numpy as np
from plot_util import plot_line, update_trace, plot_config
from config import run_config, train_config, model_config, data_config


class Plotter(object):
    """
    A plotting class to handle all things related to plotting with Visdom.

    Args:
        log_dir (string): the log directory (name of the experiment), which
                          becomes the name of the visdom environment
    """
    def __init__(self, log_dir):
        self.vis = visdom.Visdom(port=8097, env=log_dir)
        # TODO: check whether to init new plots or load handles to old plots
        self.handle_dict = self._init_plots()
        self.epoch = 1

    def _init_plots(self):
        """
        Initialize the plots. Returns a dictionary containing handles to each of
        the plots.
        """
        handle_dict = {}
        nans = np.zeros((1, 2), dtype=float)
        nans.fill(np.nan)
        n_steps = data_config['sequence_length'] - 1
        ########################################################################
        # Configuration dictionaries
        ########################################################################
        for config in [run_config, train_config, model_config, data_config]:
            plot_config(self.vis, config)
        ########################################################################
        # Total free energy, conditional log likelihood, KL divergence
        ########################################################################
        handle_dict['fe'] = plot_line(self.vis, nans, np.ones((1, 2)), legend=['Train', 'Val'],
                                      title='Total Free Energy', xlabel='Epochs',
                                      ylabel='Free Energy (Nats)', xformat='log', yformat='log')
        handle_dict['cll'] = plot_line(self.vis, nans, np.ones((1, 2)), legend=['Train', 'Val'],
                                       title='Total Conditional Log Likelihood', xlabel='Epochs',
                                       ylabel='Conditional Log Likelihood (Nats)',
                                       xformat='log', yformat='log')
        handle_dict['kl'] = plot_line(self.vis, nans, np.ones((1, 2)), legend=['Train', 'Val'],
                                      title='Total KL Divergence', xlabel='Epochs',
                                      ylabel='KL Divergence (Nats)', xformat='log', yformat='log')
        ########################################################################
        # Per step free energy, conditional log likelihood, KL divergence
        ########################################################################
        step_legend = []
        for split in ['Train', 'Val']:
            for step_num in range(1, n_steps + 1):
                step_legend.append(split + ', Step ' + str(step_num))
        handle_dict['fe_step'] = plot_line(self.vis,
                                           nans.repeat(n_steps, 1),
                                           np.ones((1, 2 * n_steps)),
                                           legend=step_legend,
                                           title='Per Step Free Energy',
                                           xlabel='Epochs',
                                           ylabel='Free Energy (Nats)',
                                           xformat='log', yformat='log')
        handle_dict['cll_step'] = plot_line(self.vis,
                                            nans.repeat(n_steps, 1),
                                            np.ones((1, 2 * n_steps)),
                                            legend=step_legend,
                                            title='Per Step Conditional Log Likelihood',
                                            xlabel='Epochs',
                                            ylabel='Conditional Log Likelihood (Nats)',
                                            xformat='log', yformat='log')
        handle_dict['kl_step'] = plot_line(self.vis,
                                           nans.repeat(n_steps, 1),
                                           np.ones((1, 2 * n_steps)),
                                           legend=step_legend,
                                           title='Per Step KL Divergence',
                                           xlabel='Epochs',
                                           ylabel='KL Divergence (Nats)',
                                           xformat='log', yformat='log')
        ########################################################################
        # Inference gradient magnitudes
        ########################################################################
        it_legend = []
        for split in ['Train', 'Val']:
            for it_num in range(run_config['inference_iterations']+1):
                it_legend.append(split + ', Iteration ' + str(it_num))
        handle_dict['mean_grad'] = plot_line(self.vis,
                                             nans.repeat(train_config['inference_iterations']+1, 1),
                                             np.ones((1, 2 * (train_config['inference_iterations']+1))),
                                             legend=it_legend,
                                             title='Mean Gradient Magnitude',
                                             xlabel='Epochs', ylabel='Mean Gradient Mag.',
                                             xformat='log', yformat='log')
        handle_dict['log_var_grad'] = plot_line(self.vis,
                                                nans.repeat(train_config['inference_iterations']+1, 1),
                                                np.ones((1, 2 * (train_config['inference_iterations']+1))),
                                                legend=it_legend,
                                                title='Log Variance Gradient Magnitude',
                                                xlabel='Epochs', ylabel='Log Variance Gradient Mag.',
                                                xformat='log', yformat='log')
        ########################################################################
        # Model parameter gradient magnitudes
        ########################################################################
        handle_dict['param_grad'] = plot_line(self.vis, nans, np.ones((1, 2)),
                                              legend=['Inf.', 'Gen.'],
                                              title='Parameter Gradient Mag.',
                                              xlabel='Epochs', ylabel='Parameter Gradient',
                                              xformat='log', yformat='log')
        ########################################################################
        # Inference improvement
        ########################################################################
        it_legend = []
        for split in ['Train', 'Val']:
            for it_num in range(1, train_config['inference_iterations']+1):
                it_legend.append(split + ', Iteration ' + str(it_num))
        handle_dict['inf_improvement'] = plot_line(self.vis,
                                                   nans.repeat(train_config['inference_iterations'], 1),
                                                   np.ones((1, 2*train_config['inference_iterations'])),
                                                   legend=it_legend,
                                                   title='Inference Improvement',
                                                   xlabel='Epochs', ylabel='Relative Improvement (%)',
                                                   xformat='log', yformat='linear')
        ########################################################################
        # Misc.
        ########################################################################
        it_legend = []
        for split in ['Train', 'Val']:
            for it_num in range(train_config['inference_iterations']+1):
                it_legend.append(split + ', Iteration ' + str(it_num))
        handle_dict['lr'] = plot_line(self.vis, nans, np.ones((1, 2)), legend=['Inf.', 'Gen.'],
                                      title='Learning Rates', xlabel='Epochs',
                                      ylabel='Learning Rate', xformat='log', yformat='log')
        handle_dict['out_log_var'] = plot_line(self.vis,
                                               nans.repeat(train_config['inference_iterations']+1, 1),
                                               np.ones((1, 2 * (train_config['inference_iterations']+1))),
                                               legend=it_legend,
                                               title='Output Log Variance',
                                               xlabel='Epochs', ylabel='Output Log Variance',
                                               xformat='log', yformat='linear')
        ########################################################################
        return handle_dict

    def plot(self, out_dict, train_val):
        """
        Function to plot all results.

        Args:
            out_dict (dict): contains numpy arrays with results to be plotted
            train_val (str): either 'Train' or 'Val', determines plotting behavior
        """
        # plot the average total and per step metrics
        metrics = [out_dict['free_energy'], out_dict['cond_log_like'], out_dict['kl_div']]
        self._plot_metrics(metrics, train_val)

        # plot the inference gradient magnitudes
        inf_grads = [out_dict['mean_grad'], out_dict['log_var_grad']]
        self._plot_inf_grads(inf_grads, train_val)

        # plot the parameter gradient magnitudes
        if train_val == 'Train':
            param_grads = [out_dict['inf_param_grad'], out_dict['gen_param_grad']]
            self._plot_param_grads(param_grads)

        # plot inference improvement
        self._plot_inf_improvement(out_dict['free_energy'], train_val)

        # plot miscellaneous results
        lr = out_dict['lr'] if 'lr' in out_dict else None
        self._plot_misc(out_dict['out_log_var'], lr, train_val)

    def _plot_metrics(self, metrics, train_val):
        """
        Plot the average total and per step metrics.

        Args:
            metrics (list): contains numpy arrays with the performance metrics
            train_val (str): either 'Train' or 'Val', determines plotting behavior
        """
        free_energy, cond_log_like, kl_div = metrics
        # total metrics, evaluated at the end of inference,summed over all steps
        update_trace(self.vis, np.array([free_energy[-1].sum()]),
                     np.array([self.epoch]).astype(int),
                     win=self.handle_dict['fe'], name=train_val)
        update_trace(self.vis, np.array([-cond_log_like[-1].sum()]),
                     np.array([self.epoch]).astype(int),
                     win=self.handle_dict['cll'], name=train_val)
        update_trace(self.vis, np.array([kl_div[-1].sum()]),
                     np.array([self.epoch]).astype(int),
                     win=self.handle_dict['kl'], name=train_val)

        # per step metrics, plot the values at the end of inference
        n_steps = data_config['sequence_length'] - 1
        for step_num in range(1, n_steps+1):
            update_trace(self.vis, np.array([free_energy[-1][step_num-1]]),
                         np.array([self.epoch]).astype(int),
                         win=self.handle_dict['fe_step'],
                         name=train_val + ', Step ' + str(step_num))
            update_trace(self.vis, np.array([-cond_log_like[-1][step_num-1]]),
                         np.array([self.epoch]).astype(int),
                         win=self.handle_dict['cll_step'],
                         name=train_val + ', Step ' + str(step_num))
            update_trace(self.vis, np.array([kl_div[-1][step_num-1]]),
                         np.array([self.epoch]).astype(int),
                         win=self.handle_dict['kl_step'],
                         name=train_val + ', Step ' + str(step_num))

    def _plot_inf_grads(self, inf_grads, train_val):
        """
        Plot inference gradient magnitudes.

        Args:
            inf_grads (list): contains numpy arrays with inference gradient mags
            train_val (str): either 'Train' or 'Val', determines plotting behavior
        """
        mean_grad, log_var_grad = inf_grads
        for it_num in range(train_config['inference_iterations']+1):
            update_trace(self.vis, np.array([mean_grad[it_num]]),
                         np.array([self.epoch]).astype(int),
                         win=self.handle_dict['mean_grad'],
                         name=train_val + ', Iteration ' + str(it_num))
            update_trace(self.vis, np.array([log_var_grad[it_num]]),
                         np.array([self.epoch]).astype(int),
                         win=self.handle_dict['log_var_grad'],
                         name=train_val + ', Iteration ' + str(it_num))

    def _plot_param_grads(self, param_grads):
        """
        Plot parameter gradient magnitudes.

        Args:
            param_grads (list): contains numpy arrays with parameter gradient mags
            train_val (str): either 'Train' or 'Val', determines plotting behavior
        """
        inf_param_grad, gen_param_grad = param_grads
        update_trace(self.vis, np.array([inf_param_grad]),
                     np.array([self.epoch]).astype(int),
                     win=self.handle_dict['param_grad'],
                     name='Inf.')
        update_trace(self.vis, np.array([gen_param_grad]),
                     np.array([self.epoch]).astype(int),
                     win=self.handle_dict['param_grad'],
                     name='Gen.')

    def _plot_inf_improvement(self, free_energy, train_val):
        """
        Plot inference improvement as a percentage of initial estimate.

        Args:
            free_energy (ndarray): shape is [n_inf_iter x n_steps]
            train_val (str): either 'Train' or 'Val', determines plotting behavior
        """
        for it_num in range(1, train_config['inference_iterations']+1):
            improvement = 100. * ((free_energy[0] - free_energy[it_num]) / free_energy[0])
            update_trace(self.vis, np.array([improvement.mean()]),
                         np.array([self.epoch]).astype(int),
                         win=self.handle_dict['inf_improvement'],
                         name=train_val + ', Iteration ' + str(it_num))

    def _plot_misc(self, out_log_var, lr, train_val):
        """
        Plot miscellaneous results.

        Args:
            out_log_var (ndarray): output log variance
            lr (list): contains inference and generation learning rates
            train_val (str): either 'Train' or 'Val', determines plotting behavior
        """
        for it_num in range(train_config['inference_iterations']+1):
            update_trace(self.vis, np.array([out_log_var[it_num].mean()]),
                         np.array([self.epoch]).astype(int),
                         win=self.handle_dict['out_log_var'],
                         name=train_val + ', Iteration ' + str(it_num))
        if lr is not None:
            update_trace(self.vis, np.array([lr[0]]), np.array([self.epoch]).astype(int),
                         win=self.handle_dict['lr'], name='Inf.')
            update_trace(self.vis, np.array([lr[1]]), np.array([self.epoch]).astype(int),
                         win=self.handle_dict['lr'], name='Gen.')

    def step(self):
        """
        Step the internal epoch counter forward one step.
        """
        self.epoch += 1

    def save(self):
        """
        Save the visdom environment.
        """
        self.vis.save([self.vis.env])
