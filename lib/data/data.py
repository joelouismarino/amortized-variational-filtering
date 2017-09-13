from __future__ import unicode_literals
import youtube_dl
import os
import torch
import scipy.misc
import skvideo.io as vid


def read_file(file_path):
    with open(file_path) as f:
        content = f.readlines()
    return content


def download_video(video_path_ext, write_path=''):
    """
    Downloads a youtube video to a specified directory.

    :param video_path_ext: the youtube extension in the URL
    :param write_path: path to the directory where the
                       video will be saved
    :return: success: whether the video was successfully
                      downloaded
    """
    assert os.path.exists(write_path), 'Write path ' + write_path + 'does not exist.'
    ydl_opts = {'quiet': True,
                'format': 'mp4',
                'outtmpl': os.path.join(write_path, '%(id)s.%(ext)s')}
    try:
        with youtube_dl.YoutubeDL(ydl_opts) as ydl:
            ydl.download(['https://www.youtube.com/watch?v=' + video_path_ext])
        success = True
    except DownloadError:
        success = False
    return success


def video_to_frames_tensor(video_file_path, n_frames):
    """
    Converts a video file into a sequence of frames in
    a PyTorch frame tensor.
    :param video_file_path: path to the video file
    :param n_frames: number of frames to extract
    :return: frames tensor: a PyTorch float tensor of frames
                            of shape [n_frames x n_channels x
                            height x width]
    """
    # todo: add image resizing capabilities so that we don't
    #       have to store a potentially massive tensor
    assert os.path.exists(video_file_path), 'Video file path does not exist.'
    frame_generator = vid.vreader(str(video_file_path))
    first_frame = next(iter(frame_generator))
    # frames_tensor initialized with shape
    # [n_frames x height x width x n_channels]
    frames_tensor = torch.zeros([n_frames] + list(first_frame.shape))
    frames_tensor[0] = torch.from_numpy(first_frame)
    for frame_num, frame in enumerate(frame_generator):
        frames_tensor[frame_num+1] = torch.from_numpy(frame)
        if frame_num == n_frames-2:
            # change to [n_frames x n_channels x height x width]
            return frames_tensor.permute(0, 3, 1, 2)


def video_to_frames_files(video_file_path, n_frames, write_dir_path):
    """
    Converts a video file into a sequence of frames and
    writes each frame to file.
    :param video_file_path: path to the video file
    :param n_frames: number of frames to extract
    :param write_dir_path: path to the directory where
                            frames are to be written
    :return: None
    """
    # todo: add image resizing capabilities so that we don't
    #       have to store a potentially massive tensor
    assert os.path.exists(video_file_path), 'Video file path does not exist.'
    frame_generator = vid.vreader(video_file_path)
    for frame_num, frame in enumerate(frame_generator):
        scipy.misc.toimage(frame, cmin=0., cmax=256.).save(os.path.join(write_dir_path, "frame%d.jpg" % frame_num))
        if frame_num == n_frames-1 and n_frames > 0:
            break


def frames_tensor_to_video(frames_tensor, video_write_path):
    """
    Convert a tensor of frames of size [n_frames x n_channels x height x width]
    into a video. Write the video to video_write_path.
    :param frames_tensor: a PyTorch tensor of shape [n_frames x n_channels x
                          height x width]
    :param video_write_path: path to write the video
    :return: None
    """
    vid.vwrite(str(video_write_path), frames_tensor.permute(0, 2, 3, 1).numpy())


def frames_files_to_video(frames_files_path, video_write_path):
    pass


def convert_to_generalized_coordinates(frames_tensor, n_orders_motion):
    """
    Converts PyTorch tensor of frames into a tensor of generalized coordinates
    with n_orders_of_motion.

    :param frames_tensor: a tensor of size [num_frames x num_channels x height x width]
                         containing the sequence of video frames
    :param n_orders_motion: number of orders of motion to extract from the video
                            (including the frames, i.e. position)
    :return:
    """

    generalized_coords_tensor = torch.zeros([n_orders_motion]+list(frames_tensor.shape))
    generalized_coords_tensor[0] = frames_tensor

    # calculate each order of motion
    for motion_order in range(1, n_orders_motion):

        frame_differences = generalized_coords_tensor[motion_order-1, 1:] - generalized_coords_tensor[motion_order-1, :-1]

        # forward differences
        generalized_coords_tensor[motion_order, 1:] = frame_differences

        # backward differences
        generalized_coords_tensor[motion_order, :-1] += frame_differences

        # average forward and backward differences at interior time steps
        generalized_coords_tensor[motion_order, 1:-1] /= 2.

    return generalized_coords_tensor


def online_generalized_coord_converter(video_file_path, n_orders_motion, video_write_path):
    pass


if __name__ == "__main__":
    video_path_extension = '3OFMvxyC-zs'
    n_frames = 200
    n_orders_motion = 5
    download_video(video_path_extension, 'temp')
    frames_tensor = video_to_frames_tensor('temp/'+video_path_extension+'.mp4', n_frames)
    generalized_tensor = convert_to_generalized_coordinates(frames_tensor, n_orders_motion)
    #for motion_order in range(n_orders_motion):
    #    video_path = 'temp/'+video_path_extension+'_order_'+str(motion_order)+'.mp4'
    #    frames_tensor_to_video(generalized_tensor[motion_order], video_path)
    del frames_tensor

    top = torch.cat((generalized_tensor[0], generalized_tensor[1]), dim=3)
    bottom = torch.cat((generalized_tensor[2], generalized_tensor[3]), dim=3)
    combined = torch.cat((top, bottom), dim=2)
    frames_tensor_to_video(combined, 'temp/'+video_path_extension+'_combined.mp4')


