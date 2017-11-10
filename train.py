from util.data.video_loader import VideoLoader
from config import train_config, model_config
import numpy as np
import matplotlib.pyplot as plt

video_loader = VideoLoader(train_config['url_file_path'], train_config['save_dir'])

img = None
for vid_reader in video_loader:
    for frame_index, frame in enumerate(vid_reader):
        frame = np.transpose(frame.numpy(), (1, 2, 0)) / 255.
        if img is None:
            img = plt.imshow(frame)
        else:
            img.set_data(frame)
        plt.pause(.05)
        plt.draw()
        if frame_index > 400:
            break
