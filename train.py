from util.data.video_loader import VideoLoader
from config import train_config, model_config
from lib.models import ConvDLVM
import numpy as np
import matplotlib.pyplot as plt

# create the data loader
video_loader = VideoLoader(train_config['url_file_path'], train_config['save_dir'])

# create the model
model = ConvDLVM(model_config)

# create the optimizers
optimizers = None

img = None
for video in video_loader:
    model.reset_state()
    for frame_index, frame in enumerate(video):
        #train(model, frame, optimizers)

        # frame = np.transpose(frame.numpy(), (1, 2, 0)) / 255.
        # if img is None:
        #     img = plt.imshow(frame)
        # else:
        #     img.set_data(frame)
        # plt.pause(.05)
        # plt.draw()
        # if frame_index > 400:
        #     break
