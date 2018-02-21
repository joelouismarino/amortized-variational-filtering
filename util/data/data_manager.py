import os
import time
from random import shuffle
from download_video import download_video


class DataManager(object):
    """
    Manages the downloading and deleteing of video files.
    Runs on a separate thread than model training.
    """
    def __init__(self, url_file_path, download_dir, queue_length=5):
        """
        url_file_path: path to the text file of YouTube URLs
        download_dir: path to the download directory
        queue_length: number of videos to maintain at any time
        """
        with open(url_file_path) as f:
            self.url_list = f.read().splitlines()
        self.num_urls = len(self.url_list)
        self.download_dir = download_dir
        # queue that contains the current video urls
        self.url_queue = []
        self.queue_length = queue_length
        self.stop_loop = False

    def add(self, url):
        # downloads a video and adds the URL to the queue
        success = download_video(url, self.download_dir)
        if success:
            self.url_queue.insert(0, url)
        return success

    def remove(self, n_remove=1):
        # removes the oldest n_remove URLs from the queue
        if len(self.url_queue) >= n_remove:
            for _ in range(n_remove):
                url = self.url_queue.pop()
                os.remove(os.path.join(self.download_dir, url + '.mp4'))
            return True
        return False

    def run(self, shuffle_urls=True):
        # loop through the urls, adding to the queue
        url_index = 0
        if shuffle_urls:
            shuffle(self.url_list)
        while True:
            if url_index >= self.num_urls:
                url_index = 0
                if shuffle_urls:
                    shuffle(self.url_list)
            if len(self.url_queue) < self.queue_length:
                success = False
                while not success:
                    current_url = self.url_list[url_index]
                    success = self.add(current_url)
                    url_index += 1
            if self.stop_loop:
                break
            time.sleep(1)

    def stop(self):
        # stops the url loop, cleans save directory
        self.stop_loop = True
        self.remove_all()

    def remove_all(self):
        # removes all videos currently in the queue
        while len(self.url_queue) > 0:
            self.remove()
