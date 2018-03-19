from config import run_config, train_config, data_config, model_config
from util.logging import init_log, load_checkpoint, save_checkpoint
from util.plotting import init_plot
from util.data.load_data import load_data
from lib.models import load_model
from util.optimizers import load_opt_sched
from util.train_val import train, validate

# initialize logging and plotting
log_dir = init_log(run_config)
init_plot(log_dir)

# load the data
train_data, val_data, test_data = load_data(data_config, run_config)

# load the model, optimizers
if run_config['resume_path']:
    model, optimizers, schedulers = load_checkpoint(run_config['resume_path'])
else:
    model = load_model(model_config)
    optimizers, schedulers = load_opt_sched(train_config, model)

# train the model
while True:
    train(train_data, model, optimizers, schedulers)
    if val_data:
        validate(val_data, model)
    save_checkpoint(model, optimizers, schedulers)




#
# # create the video loader
# # video_loader = VideoLoader(data_config, train_config['batch_size'])
#
# # create the model
# model = ConvDLVM(model_config)
#
# model = torch.load('model.ckpt', map_location=lambda storage, loc: storage)
#
# if train_config['cuda_device'] is not None:
#     model = model.cuda(train_config['cuda_device'])
#
#
#
# # create the optimizers
# enc_opt, dec_opt = get_optimizers(train_config, model)
#
# movie_img = None
# recon_img = None
#
# print 'Starting training loop...'
# for video_batch in video_loader:
#
#     # tic = time.time()
#
#     output_dims = next(iter(video_batch)).shape
#     data_size = output_dims[1] * output_dims[2] * output_dims[3]
#     model.reinitialize_variables(output_dims)
#
#     # toc = time.time()
#     # print '*' * 50
#     # print 'Variable Initialization Time : ' + str(toc - tic)
#     # print '*' * 50
#
#     print 'Starting video loop...'
#     for frame_index, frame_tensor in enumerate(video_batch):
#
#         frame_tic = time.time()
#
#         frame_tensor = torch.from_numpy(frame_tensor)
#         if train_config['cuda_device'] is not None:
#             frame_tensor = frame_tensor.cuda(train_config['cuda_device'])
#         frame_tensor = Variable(frame_tensor / 255.)
#
#         # variable_toc = time.time()
#         # print '*' * 50
#         # print 'Convert data to PyTorch Time : ' + str(variable_toc - frame_tic)
#         # print '*' * 50
#
#
#         # print 'Time Step: ' + str(frame_index)
#         # torch.cuda.empty_cache()
#         enc_opt.zero_grad()
#         for _ in range(train_config['inference_iterations']):
#
#             # elbo_tic = time.time()
#
#             loss = -model.elbo(frame_tensor, averaged=True) / data_size
#
#             # elbo_toc = time.time()
#             # print '*' * 50
#             # print 'ELBO evaluation Time : ' + str(elbo_toc - elbo_tic)
#             # print '*' * 50
#
#             # backward_tic = time.time()
#
#             loss.backward(retain_graph=True)
#
#             # backward_toc = time.time()
#             # print '*' * 50
#             # print 'Backward Time : ' + str(backward_toc - backward_tic)
#             # print '*' * 50
#
#             # inference_tic = time.time()
#
#             model.infer(frame_tensor-0.5)
#
#             # inference_toc = time.time()
#             # print '*' * 50
#             # print 'Inference Time : ' + str(inference_toc - inference_tic)
#             # print '*' * 50
#
#             # generation_tic = time.time()
#
#             model.generate()
#
#             # generation_toc = time.time()
#             # print '*' * 50
#             # print 'Generation Time : ' + str(generation_toc - generation_tic)
#             # print '*' * 50
#
#         dec_opt.zero_grad()
#         # loss = -model.elbo(frame_tensor, averaged=True) / (64. * 64. * 3.)
#
#         # loss_tic = time.time()
#
#         elbo, cond_log_like, kl = model.losses(frame_tensor, averaged=True)
#         # loss = -model.elbo(frame_tensor, averaged=True) / data_size
#
#         # loss_toc = time.time()
#         # print '*' * 50
#         # print 'Loss Time : ' + str(loss_toc - loss_tic)
#         # print '*' * 50
#
#         # loss.backward(retain_graph=True)
#
#         # backward_tic = time.time()
#
#         (-elbo / data_size).backward(retain_graph=True)
#         # loss.backward(retain_graph=True)
#
#         # backward_toc = time.time()
#         # print '*' * 50
#         # print 'Backward 2 Time : ' + str(backward_toc - backward_tic)
#         # print '*' * 50
#
#         # (-elbo).backward(retain_graph=True)
#         # print 'Loss: ' + str(loss.data[0])
#         print 'ELBO: ' + str(elbo.data[0]) + ' Cond Log Like: ' + str(cond_log_like.data[0]) + ' KL: ' + str(kl[0].data[0])
#
#         # param_ave_tic = time.time()
#
#         for param in model.inference_model_parameters():
#             param.grad /= train_config['inference_iterations']
#
#         # param_ave_toc = time.time()
#         # print '*' * 50
#         # print 'Parameter Averaging Time : ' + str(param_ave_toc - param_ave_tic)
#         # print '*' * 50
#
#         # step_tic = time.time()
#
#         enc_opt.step()
#         dec_opt.step()
#
#         # step_toc = time.time()
#         # print '*' * 50
#         # print 'Optimizer Step Time : ' + str(step_toc - step_tic)
#         # print '*' * 50
#
#         # plt_tic = time.time()
#         #
#         plt.figure(1)
#         movie_frame = np.transpose(frame_tensor[0].data.cpu().numpy(), (1, 2, 0))
#         if movie_img is None:
#             movie_img = plt.imshow(movie_frame)
#         else:
#             movie_img.set_data(movie_frame)
#         plt.pause(.05)
#         plt.draw()
#
#         plt.figure(2)
#         recon_frame = np.transpose(model.output_dist.mean[0, 0].data.cpu().numpy(), (1, 2, 0))
#         if recon_img is None:
#             recon_img = plt.imshow(recon_frame)
#         else:
#             recon_img.set_data(recon_frame)
#         plt.pause(.01)
#         plt.draw()
#         #
#         # plt_toc = time.time()
#         # print '*' * 50
#         # print 'Plot Time : ' + str(plt_toc - plt_tic)
#         # print '*' * 50
#         #
#         # step_tic = time.time()
#         #
#         model.step()
#         #
#         # step_toc = time.time()
#         # print '*' * 50
#         # print 'Model Step Time : ' + str(step_toc - step_tic)
#         # print '*' * 50
#
#         frame_toc = time.time()
#         # print '*' * 50
#         # print '*' * 50
#         print 'TOTAL FRAME TIME : ' + str(frame_toc - frame_tic)
#         # print '*' * 50
#         # print '*' * 50
#
#         if frame_index > 600:
#             break
