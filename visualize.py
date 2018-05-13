import torch
from config import run_config, train_config, data_config, model_config
from util.logging import Logger
from util.plotting import Plotter
from util.data.load_data import load_data
from lib.models import load_model
from util.optimization import load_opt_sched
from util.train_val import visualize

import matplotlib.pyplot as plt
import numpy as np
import cPickle
from util.plotting.audio_util import write_audio

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=str(run_config['cuda_device'])
# torch.cuda.set_device(run_config['cuda_device'])
torch.cuda.set_device(0)

# initialize logging
logger = Logger(run_config)

# load the data
train_data, val_data, test_data = load_data(data_config, train_config['batch_size'])
if val_data is None:
    val_data = test_data

data = train_data

# load the model, optimizers
print('Loading model...')
model = load_model(model_config)

assert run_config['resume_path'] is not None, 'Model must be resumed from checkpoint.'

if run_config['resume_path']:
    print('Resuming checkpoint ' + run_config['resume_path'])
    model = logger.load_best(model)

print('Putting the model on the GPU...')

model.cuda()
out = visualize(data, model)


################################################################################
## plot data, predictions, and reconstructions

batch_index = 10
plt.figure()

data_shape = out['output_mean'].shape
if len(data_shape) == 6:
    # image data
    batch_size, n_inf_iter, n_steps, c, h, w = data_shape

    # visualize the output mean
    for step in range(1, n_steps+1):

        # prediction
        plt.subplot(3, n_steps, step)
        prediction = out['output_mean'][batch_index, 0, step-1]
        if c == 3:
            # RGB
            plt.imshow(prediction.transpose(1, 2, 0))
        else:
            # B&W
            plt.imshow(prediction[0], cmap='gray')
        plt.title('t = ' + str(step))
        plt.axis('off')

        # reconstruction
        plt.subplot(3, n_steps, step + n_steps)
        recon = out['output_mean'][batch_index, 1, step-1]
        if c == 3:
            # RGB
            plt.imshow(recon.transpose(1, 2, 0))
        else:
            # B&W
            plt.imshow(recon[0], cmap='gray')
        plt.axis('off')

        # data
        plt.subplot(3, n_steps, step + 2 * n_steps)
        data = out['data'][batch_index, step-1]
        if c == 3:
            # RGB
            plt.imshow(data.transpose(1, 2, 0))
        else:
            # B&W
            plt.imshow(data[0], cmap='gray')
        plt.axis('off')


else:
    # non-image data
    batch_size, n_inf_iter, n_steps, m = data_shape

    # audio
    out_mean = np.concatenate([out['output_mean'][:, :, i, :] for i in range(n_steps)], axis=2)
    out_log_var = np.concatenate([out['output_log_var'][:, :, i, :] for i in range(n_steps)], axis=2)
    data = np.concatenate([out['data'][:, i, :] for i in range(n_steps)], axis=1)

    kl_div = np.tile(out['kl_div'].reshape(batch_size, n_inf_iter, n_steps, 1), [1, 1, 1, m]).reshape(batch_size, n_inf_iter, -1)
    cond_log_like = np.tile(out['cond_log_like'].reshape(batch_size, n_inf_iter, n_steps, 1), [1, 1, 1, m]).reshape(batch_size, n_inf_iter, -1)

    min_val = min(np.min(out_mean[batch_index]), np.min(data[batch_index]))
    max_val = max(np.max(out_mean[batch_index]), np.max(data[batch_index]))

    # prediction
    plt.subplot(3, 1, 1)
    pred_mean = out_mean[batch_index, 0]
    plt.plot(pred_mean)
    ax = plt.gca()
    pred_std = np.exp(0.5 * out_log_var[batch_index, 0])
    ax.fill_between(range(out_mean.shape[2]), pred_mean - pred_std, pred_mean + pred_std, alpha=0.5)
    plt.ylim((min_val, max_val))
    plt.axis('off')

    # reconstruction
    plt.subplot(3, 1, 1)
    recon_mean = out_mean[batch_index, 1]
    plt.plot(recon_mean)
    ax = plt.gca()
    recon_std = np.exp(0.5 * out_log_var[batch_index, 1])
    ax.fill_between(range(out_mean.shape[2]), recon_mean - recon_std, recon_mean + recon_std, alpha=0.5)
    plt.ylim((min_val, max_val))
    plt.axis('off')

    plt.subplot(3, 1, 3)
    error = (out_mean[batch_index, 1] - data[batch_index])
    plt.plot(error)
    # plt.ylim((min_val, max_val))
    plt.axis('off')



    data_mean, data_std = cPickle.load(open(os.path.join(data_config['data_path'], 'timit', 'statistics.p'), 'r'))

    original_audio = data[batch_index] * data_std + data_mean
    write_audio(original_audio, '/home/joe/vrnn_original_audio.wav')
    pred_audio = pred_mean * data_std + data_mean
    write_audio(pred_audio, '/home/joe/vrnn_pred_audio.wav')
    recon_audio = recon_mean * data_std + data_mean
    write_audio(recon_audio, '/home/joe/vrnn_recon_audio.wav')



    # plt.subplot(3, 1, 3)
    # pw_error = (out_mean[batch_index, 1] - data[batch_index]) / (recon_std **2)
    # plt.plot(pw_error)
    # # plt.ylim((min_val, max_val))
    # plt.axis('off')

    # plt.subplot(3, 1, 3)
    # plt.plot(cond_log_like[batch_index, 1])
    # plt.axis('off')

    # plt.subplot(3, 1, 3)
    # diff = recon_mean - pred_mean
    # plt.plot(diff)
    # plt.axis('off')

    # data
    plt.subplot(3, 1, 2)
    plt.plot(data[batch_index])
    plt.ylim((min_val, max_val))
    plt.axis('off')




plt.figure()
plt.plot(pred_mean, 'r')
ax = plt.gca()
pred_std = np.exp(0.5 * out_log_var[batch_index, 0])
ax.fill_between(range(out_mean.shape[2]), pred_mean - pred_std, pred_mean + pred_std, alpha=0.5, facecolor='red')
plt.plot(data[batch_index], 'b', alpha=0.7)
plt.axis('off')

"""
log_var = out['output_log_var'][batch_index]

# visualize output mean
for step in range(1, n_steps+1):

    # prediction
    plt.subplot(3, n_steps, step)
    plt.imshow(out['output_log_var'][batch_index, 0, step-1].transpose(1, 2, 0))
    plt.title('t = ' + str(step))
    plt.axis('off')

    # reconstruction
    plt.subplot(3, n_steps, step + n_steps)
    plt.imshow(out['output_log_var'][batch_index, 1, step-1].transpose(1, 2, 0))
    plt.axis('off')

    # data
    plt.subplot(3, n_steps, step + 2 * n_steps)
    plt.imshow(out['data'][batch_index, step-1].transpose(1, 2, 0))
    plt.axis('off')


data = out['data'][batch_index, :, 0]
pred_mean = out['output_mean'][batch_index, 0, :, 0]
pred_log_var = out['output_log_var'][batch_index, 0, :, 0]
recon_mean = out['output_mean'][batch_index, 1, :, 0]
recon_log_var = out['output_log_var'][batch_index, 1, :, 0]

pred_error = np.absolute((pred_mean - data) / np.exp(pred_log_var))
recon_error = np.absolute((recon_mean - data) / np.exp(recon_log_var))

max_value = max(np.max(pred_error), np.max(recon_error))

pred_error /= max_value
recon_error /= max_value

# visualize prevision weighted errors
for step in range(1, n_steps+1):

    # prediction
    plt.subplot(3, n_steps, step)
    plt.imshow(pred_error[step-1], cmap='gray')
    plt.title('t = ' + str(step))
    plt.axis('off')

    # reconstruction
    plt.subplot(3, n_steps, step + n_steps)
    plt.imshow(recon_error[step-1], cmap='gray')
    plt.axis('off')

    # data
    plt.subplot(3, n_steps, step + 2 * n_steps)
    plt.imshow(out['data'][batch_index, step-1, 0], cmap='gray')
    plt.axis('off')


if False:

    ################################################################################

    ## plot data, predictions, and reconstructions
    batch_size, n_inf_iter, n_steps, c, h, w = out['output_mean'].shape

    batch_index = 10

    plt.figure()

    # visualize output mean
    for step in range(1, n_steps+1):

        # prediction
        plt.subplot(3, n_steps, step)
        plt.imshow(out['output_mean'][batch_index, 0, step-1, 0], cmap='gray')
        plt.title('t = ' + str(step))
        plt.axis('off')

        # reconstruction
        plt.subplot(3, n_steps, step + n_steps)
        plt.imshow(out['output_mean'][batch_index, 1, step-1, 0], cmap='gray')
        plt.axis('off')

        # data
        plt.subplot(3, n_steps, step + 2 * n_steps)
        plt.imshow(out['data'][batch_index, step-1, 0], cmap='gray')
        plt.axis('off')

    # visualize output log_variance
    for step in range(1, n_steps+1):

        # prediction
        plt.subplot(3, n_steps, step)
        plt.imshow(out['output_log_var'][batch_index, 0, step-1, 0], cmap='gray')
        plt.title('t = ' + str(step))
        plt.axis('off')

        # reconstruction
        plt.subplot(3, n_steps, step + n_steps)
        plt.imshow(out['output_log_var'][batch_index, 1, step-1, 0], cmap='gray')
        plt.axis('off')

        # data
        plt.subplot(3, n_steps, step + 2 * n_steps)
        plt.imshow(out['data'][batch_index, step-1, 0], cmap='gray')
        plt.axis('off')


    data = out['data'][batch_index, :, 0]
    pred_mean = out['output_mean'][batch_index, 0, :, 0]
    pred_log_var = out['output_log_var'][batch_index, 0, :, 0]
    recon_mean = out['output_mean'][batch_index, 1, :, 0]
    recon_log_var = out['output_log_var'][batch_index, 1, :, 0]

    pred_error = np.absolute((pred_mean - data) / np.exp(pred_log_var))
    recon_error = np.absolute((recon_mean - data) / np.exp(recon_log_var))

    max_value = max(np.max(pred_error), np.max(recon_error))

    pred_error /= max_value
    recon_error /= max_value

    # visualize prevision weighted errors
    for step in range(1, n_steps+1):

        # prediction
        plt.subplot(3, n_steps, step)
        plt.imshow(pred_error[step-1], cmap='gray')
        plt.title('t = ' + str(step))
        plt.axis('off')

        # reconstruction
        plt.subplot(3, n_steps, step + n_steps)
        plt.imshow(recon_error[step-1], cmap='gray')
        plt.axis('off')

        # data
        plt.subplot(3, n_steps, step + 2 * n_steps)
        plt.imshow(out['data'][batch_index, step-1, 0], cmap='gray')
        plt.axis('off')
"""
