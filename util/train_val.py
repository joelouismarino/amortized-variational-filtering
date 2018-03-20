from torch.autograd import Variable
from logging import log_wrapper
from plotting import plot_wrapper


@plot_wrapper
@log_wrapper
def train(data, model, optimizers, schedulers):
    """
    Function to train the model on data and update using optimizers and schedulers.

    Args:
        data (DataLoader): a data loader that provides batches of sequence data
        model (LatentVariableModel): model to train
        optimizers (tuple): inference and generative optimizers
        schedulers (tuple): inference and generative optimizer schedulers
    """
    inf_opt, gen_opt = optimizers
    inf_sched, gen_sched = schedulers

    for batch_ind, batch in enumerate(data):
        model.re_init()
        for step_ind, step_batch in enumerate(batch):
            model.infer(Variable(step_batch))
            model.generate()
            model.step()


@plot_wrapper
@log_wrapper
def validate(data, model):
    """
    Function to validate the model on data and update using optimizers and schedulers.

    Args:
        data (DataLoader): a data loader that provides batches of sequence data
        model (LatentVariableModel): model to train
    """
    for batch_ind, batch in enumerate(data):
        model.re_init()
        for step_ind, step_batch in enumerate(batch):
            model.infer(Variable(step_batch))
            model.generate()
            model.step()
