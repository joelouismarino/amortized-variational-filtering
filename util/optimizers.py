import torch.optim as opt
from torch.optim.lr_scheduler import ExponentialLR


def load_opt_sched(train_config, model):

    encoder_params = model.inference_model_parameters()
    decoder_params = model.generative_model_parameters()

    opt_name = train_config['optimizer'].lower().replace('_', '').strip()
    if opt_name == 'sgd':
        optimizer = opt.SGD
    elif opt_name == 'rmsprop':
        optimizer = opt.RMSprop
    elif opt_name == 'adam':
        optimizer = opt.Adam

    enc_opt = optimizer(encoder_params, lr=train_config['encoder_learning_rate'])
    dec_opt = optimizer(decoder_params, lr=train_config['decoder_learning_rate'])

    enc_sched = ExponentialLR(enc_opt, 0.999)
    dec_sched = ExponentialLR(dec_opt, 0.999)

    return (enc_opt, dec_opt), (enc_sched, dec_sched)
