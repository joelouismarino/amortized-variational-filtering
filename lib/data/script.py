import os
import torch
import torchvision
import numpy as np
from download_video import download_video
from data import video_to_frames_tensor
from video_reader import VideoReader

import matplotlib.pyplot as plt

url = 'eZh0R6uB5Zc'
write_path = 'temp/'

# download_video(url, write_path)
video_str = os.path.join(write_path, url + '.mp4')
vid_reader = VideoReader(video_str, 3)

img = None
for coords_frame in vid_reader:
    grid_image = torchvision.utils.make_grid(coords_frame, normalize=True)
    grid_image = np.transpose(grid_image.numpy(), (1, 2, 0))
    if img is None:
        img = plt.imshow(grid_image)
    else:
        img.set_data(grid_image)
    plt.pause(.1)
    plt.draw()


################################################################################
# INSPECT THE DERIVATIVES
# load the frames tensor
n_frames = 100
spatial_location = [200, 200]
frames_tensor = video_to_frames_tensor(video_str, n_frames)
frames_tensor = frames_tensor.numpy()

# forward finite differences
fwd_velocity_tensor = np.zeros(frames_tensor.shape)
fwd_velocity_tensor[1:] = frames_tensor[1:] - frames_tensor[:-1]

# center fiinite differences
ctr_velocity_tensor = np.zeros(frames_tensor.shape)
ctr_velocity_tensor[1:-1] = (frames_tensor[2:] - frames_tensor[:-2]) / 2

# plot the frames and velocities
plt.figure(1)
plt.plot(frames_tensor[:, 0, spatial_location[0], spatial_location[1]], 'r')
plt.plot(frames_tensor[:, 1, spatial_location[0], spatial_location[1]], 'g')
plt.plot(frames_tensor[:, 2, spatial_location[0], spatial_location[1]], 'b')

plt.figure(2)
plt.plot(fwd_velocity_tensor[:, 0, spatial_location[0], spatial_location[1]], 'r')
plt.plot(fwd_velocity_tensor[:, 1, spatial_location[0], spatial_location[1]], 'g')
plt.plot(fwd_velocity_tensor[:, 2, spatial_location[0], spatial_location[1]], 'b')

plt.figure(3)
plt.plot(ctr_velocity_tensor[:, 0, spatial_location[0], spatial_location[1]], 'r')
plt.plot(ctr_velocity_tensor[:, 1, spatial_location[0], spatial_location[1]], 'g')
plt.plot(ctr_velocity_tensor[:, 2, spatial_location[0], spatial_location[1]], 'b')

plt.show()

norm_ctr_velocity = (ctr_velocity_tensor - ctr_velocity_tensor.min()) / (ctr_velocity_tensor.max() - ctr_velocity_tensor.min())
norm_ctr_velocity = np.transpose(norm_ctr_velocity,  (0, 2, 3, 1))

norm_fwd_velocity = (fwd_velocity_tensor - fwd_velocity_tensor.min()) / (fwd_velocity_tensor.max() - fwd_velocity_tensor.min())
norm_fwd_velocity = np.transpose(norm_fwd_velocity,  (0, 2, 3, 1))

img = None
for frame_num in range(100):
    frame = norm_fwd_velocity[frame_num]
    if img is None:
        img = plt.imshow(frame)
    else:
        img.set_data(frame)
    plt.pause(.1)
    plt.draw()
