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
            for step_num in range(1, data_config['sequence_length']+1):
                step_legend.append(split + ', Step ' + str(step_num))
        handle_dict['fe_step'] = plot_line(self.vis,
                                           nans.repeat(data_config['sequence_length'], 1),
                                           np.ones((1, 2 * data_config['sequence_length'])),
                                           legend=step_legend,
                                           title='Per Step Free Energy',
                                           xlabel='Epochs',
                                           ylabel='Free Energy (Nats)',
                                           xformat='log', yformat='log')
        handle_dict['cll_step'] = plot_line(self.vis,
                                            nans.repeat(data_config['sequence_length'], 1),
                                            np.ones((1, 2 * data_config['sequence_length'])),
                                            legend=step_legend,
                                            title='Per Step Conditional Log Likelihood',
                                            xlabel='Epochs',
                                            ylabel='Conditional Log Likelihood (Nats)',
                                            xformat='log', yformat='log')
        handle_dict['kl_step'] = plot_line(self.vis,
                                           nans.repeat(data_config['sequence_length'], 1),
                                           np.ones((1, 2 * data_config['sequence_length'])),
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
            for it_num in range(1, run_config['inference_iterations']+1):
                it_legend.append(split + ', Iteration ' + str(it_num))
        handle_dict['mean_grad'] = plot_line(self.vis,
                                             nans.repeat(run_config['inference_iterations'], 1),
                                             np.ones((1, 2 * run_config['inference_iterations'])),
                                             legend=it_legend,
                                             title='Mean Gradient Mag.',
                                             xlabel='Epochs', ylabel='Mean Gradient',
                                             xformat='log', yformat='log')
        handle_dict['log_var_grad'] = plot_line(self.vis,
                                                nans.repeat(run_config['inference_iterations'], 1),
                                                np.ones((1, 2 * run_config['inference_iterations'])),
                                                legend=it_legend,
                                                title='Log Variance Gradient Mag.',
                                                xlabel='Epochs', ylabel='Log Variance Gradient',
                                                xformat='log', yformat='log')
        ########################################################################
        # Learning gradient magnitudes
        ########################################################################
        handle_dict['param_grad'] = plot_line(self.vis, nans, np.ones((1, 2)),
                                              legend=['Inf.', 'Gen.'],
                                              title='Parameter Gradient Mag.',
                                              xlabel='Epochs', ylabel='Parameter Gradient',
                                              xformat='log', yformat='log')
        ########################################################################
        # Inference improvement
        ########################################################################
        # TODO

        ########################################################################
        # Misc.
        ########################################################################
        handle_dict['lr'] = plot_line(self.vis, nans, np.ones((1, 2)), legend=['Inf.', 'Gen.'],
                                      title='Learning Rates', xlabel='Epochs',
                                      ylabel='Learning Rate', xformat='log', yformat='log')
        handle_dict['out_log_var'] = plot_line(self.vis, nans, np.ones((1, 2)),
                                               legend=['Train', 'Val'], title='Output Log Variance',
                                               xlabel='Epochs', ylabel='Log Variance',
                                               xformat='log', yformat='linear')
        ########################################################################
        return handle_dict

    def plot(self, out_dict, train_val):
        """
        Function to plot all results.
        """
        # plot the average total and per step metrics
        metrics = [out_dict['free_energy'], out_dict['cond_log_like'], out_dict['kl_div']]
        self._plot_metrics(metrics, train_val)

        # plot the inference gradient magnitudes
        # inf_grads = [out_dict['mean_grad'], out_dict['log_var_grad']]
        # self._plot_inf_grads(inf_grads, train_val)

        # plot the parameter gradient magnitudes
        # if optimizers is not None:
        #     plot_param_grads(out_dict['avg_param_grad_mags'], epoch, handle_dict)
        #     plot_opt_lr(optimizers, epoch, handle_dict)

        # plot inference improvement
        # self._plot_inf_improvement()

        # plot miscellaneous results
        lr = out_dict['lr'] if 'lr' in out_dict else None
        self._plot_misc(out_dict['out_log_var'], lr, train_val)

    def _plot_metrics(self, metrics, train_val):
        """
        Plot the average total and per step metrics.
        """
        free_energy, cond_log_like, kl_div = metrics
        # total metrics
        update_trace(self.vis, np.array([free_energy.sum(axis=1).mean()]),
                     np.array([self.epoch]).astype(int),
                     win=self.handle_dict['fe'], name=train_val)
        update_trace(self.vis, np.array([-cond_log_like.sum(axis=1).mean()]),
                     np.array([self.epoch]).astype(int),
                     win=self.handle_dict['cll'], name=train_val)
        update_trace(self.vis, np.array([kl_div.sum(axis=1).mean()]),
                     np.array([self.epoch]).astype(int),
                     win=self.handle_dict['kl'], name=train_val)

        # per step metrics
        for step_num in range(1, data_config['sequence_length']+1):
            update_trace(self.vis, np.array([free_energy.mean(axis=0)[step_num-1]]),
                         np.array([self.epoch]).astype(int),
                         win=self.handle_dict['fe_step'],
                         name=train_val + ', Step ' + str(step_num))
            update_trace(self.vis, np.array([-cond_log_like.mean(axis=0)[step_num-1]]),
                         np.array([self.epoch]).astype(int),
                         win=self.handle_dict['cll_step'],
                         name=train_val + ', Step ' + str(step_num))
            update_trace(self.vis, np.array([kl_div.mean(axis=0)[step_num-1]]),
                         np.array([self.epoch]).astype(int),
                         win=self.handle_dict['kl_step'],
                         name=train_val + ', Step ' + str(step_num))

    def _plot_inf_grads(self, inf_grads, train_val):
        """
        Plot inference gradient magnitudes.
        """
        mean_grad, log_var_grad = inf_grads

    def _plot_param_grads(self, param_grads, train_val):
        """
        Plot parameter gradient magnitudes.
        """
        pass

    def _plot_inf_improvement(self):
        """
        Plot inference improvement.
        """
        pass

    def _plot_misc(self, out_log_var, lr, train_val):
        """
        Plot miscellaneous results.
        """
        update_trace(self.vis, np.array([out_log_var.mean()]), np.array([self.epoch]).astype(int),
                     win=self.handle_dict['out_log_var'], name=train_val)
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
