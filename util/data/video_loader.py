import os
import time
import threading
from data_manager import DataManager
from video_reader import VideoReader


class VideoLoader(object):
    """
    Converts a text file of YouTube URLs into batched video readers.
    """
    def __init__(self, data_config, batch_size):
        """
        data_config: a dictionary containing
            url_file_path: path to the text file of YouTube URLs
            download_dir: path to the download directory
            transform: whether to crop/resize the frame_tensor
            resize: resize dimensions for the frames (h, w)
        batch_size: the number of videos to read per batch
        """
        self.transform = data_config['transform']
        self.resize = data_config['resize']
        self.batch_size = batch_size

        # create the data_manager, start downloading videos
        self.url_file_path = data_config['url_file_path']
        self.data_save_dir = data_config['data_save_dir']
        self.data_manager = DataManager(self.url_file_path, self.data_save_dir, 2 * self.batch_size)
        data_thread = threading.Thread(target=self.data_manager.run)
        data_thread.start()

    def __iter__(self):
        while True:
            self.data_manager.remove(self.batch_size) # remove the previous batch of URLs
            # confirm that we have at least batch_size files saved to disk
            while len(self.data_manager.url_queue) < self.batch_size:
                time.sleep(1)
            # retrieve the next batch_size URLs and create a video reader
            urls = self.data_manager.url_queue[-self.batch_size:]
            urls = [os.path.join(self.data_save_dir, url + '.mp4') for url in urls]
            vid_reader = VideoReader(urls, self.transform, self.resize)
            yield vid_reader

    def close(self):
        # clean up the data manager
        self.data_manager.stop()
