from torch.autograd import Variable


def train(data, model, optimizers):

    for batch_ind, batch in enumerate(data):
        model.re_init()
        for step_ind, step_batch in enumerate(batch):
            model(Variable(step_batch))


def validate(data, model):
    
    for batch_ind, batch in enumerate(data):
        model.re_init()
        for step_ind, step_batch in enumerate(batch):
            model(Variable(step_batch))
