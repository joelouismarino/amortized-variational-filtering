import os
from PIL import Image
from torch.utils.data import Dataset


class KTHActions(Dataset):
    """
    Dataset object for KTH actions dataset. The dataset must be stored
    with each video (action sequence) in a separate directory:
        /path
            /person01_walking_d1_0
                /0.png
                /1.png
                /...
            /person01_walking_d1_1
                /...
    """
    def __init__(self, path, transform=None):
        assert os.path.exists(path), 'Invalid path to KTH actions data set: ' + path
        self.path = path
        self.transform = transform
        self.video_list = os.listdir(self.path)

    def __getitem__(self, ind):
        # load the images from the ind directory to get list of PIL images
        img_names = os.listdir(os.path.join(self.path, self.video_list[ind]))
        imgs = [Image.open(os.path.join(self.path, self.video_list[ind], i)) for i in img_names]
        if self.transform is not None:
            # apply the image/video transforms
            imgs = self.transform(imgs)
        return imgs

    def __len__(self):
        # returns the total number of videos
        return len(os.listdir(self.path))
