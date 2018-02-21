import torch.optim as opt
# from torch.optim.lr_scheduler import ReduceLROnPlateau


def get_optimizers(train_config, model):

    encoder_params = model.inference_model_parameters()
    enc_opt = opt.Adam(encoder_params, lr=train_config['encoder_learning_rate'])

    decoder_params = model.generative_model_parameters()
    dec_opt = opt.Adam(decoder_params, lr=train_config['decoder_learning_rate'])

    return enc_opt, dec_opt
