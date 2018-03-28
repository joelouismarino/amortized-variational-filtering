from torch.autograd import Variable
from config import run_config
from logging import log_wrapper
from plotting import plot_wrapper


# @plot_wrapper
# @log_wrapper
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

    # loop over training examples
    for batch_ind, batch in enumerate(data):

        # re-initialize the model from the data
        batch = Variable(batch.transpose(0, 1).cuda())
        model.re_init(batch[0])
        inf_opt.zero_grad(); gen_opt.zero_grad()

        # loop over sequence steps
        for step_ind, step_batch in enumerate(batch[1:]):

            # form a prediction, get gradients/errors
            # model.generate(gen=True)
            # model.free_energy(step_batch).backward(retain_graph=True)

            # loop over inference iterations
            for inf_it in range(run_config['inference_iterations']):
                model.infer(step_batch)
                model.generate()
                # model.free_energy(step_batch).backward(retain_graph=True)
                # inf_opt.step()
                # inf_opt.zero_grad()
            # gen_opt.step()
            # gen_opt.zero_grad()

            # form the prior on the next step
            model.step()

            print(batch_ind, step_ind)
            print(model.free_energy(step_batch))
        import ipdb; ipdb.set_trace()


# @plot_wrapper
# @log_wrapper
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
