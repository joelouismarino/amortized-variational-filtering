from __future__ import unicode_literals
import youtube_dl
import os


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
    except:
        success = False
    return success
