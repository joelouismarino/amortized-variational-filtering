import os
import time
import threading
from data_manager import DataManager
from video_reader import VideoReader


class VideoLoader(object):
    """
    Converts a text file of YouTube URLs into video readers.
    """
    def __init__(self, url_file_path, download_dir, transforms=None):
        """
        url_file_path: path to the text file of YouTube URLs
        download_dir: path to the download directory
        transforms: image transformations to perform
        """
        # create the data_manager, start downloading videos
        self.url_file_path = url_file_path
        self.download_dir = download_dir
        self.data_manager = DataManager(self.url_file_path, self.download_dir)
        data_thread = threading.Thread(target=self.data_manager.run)
        data_thread.start()

        self.transforms = transforms

    def __iter__(self):
        while True:
            self.data_manager.remove()
            # confirm that we have a file saved to disk
            while len(self.data_manager.url_queue) == 0:
                time.sleep(1)
            # retrieve the next URL and create a video reader
            url = self.data_manager.url_queue[-1]
            print 'Found URL ' + url
            url = os.path.join(self.download_dir, url + '.mp4')
            print 'Full path: ' + url
            vid_reader = VideoReader(url, transforms=self.transforms)
            yield vid_reader

    def close(self):
        # clean up the data manager
        self.data_manager.stop()
