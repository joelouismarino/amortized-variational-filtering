from optimizer import Optimizer
from torch.optim.lr_scheduler import ExponentialLR


def load_opt_sched(train_config, model):

    inf_params = model.inference_parameters()
    gen_params = model.generative_parameters()

    inf_opt = Optimizer(train_config['optimizer'], inf_params,
                        lr=train_config['inference_learning_rate'])
    inf_sched = ExponentialLR(inf_opt.opt, 0.999)

    gen_opt = Optimizer(train_config['optimizer'], gen_params,
                        lr=train_config['generation_learning_rate'])
    gen_sched = ExponentialLR(gen_opt.opt, 0.999)

    return (inf_opt, gen_opt), (inf_sched, gen_sched)


def load_sched(optimizers, last_epoch):
    inf_opt, gen_opt = optimizers
    inf_sched = ExponentialLR(inf_opt.opt, 0.999, last_epoch=last_epoch)
    gen_sched = ExponentialLR(gen_opt.opt, 0.999, last_epoch=last_epoch)
    return (inf_sched, gen_sched)
