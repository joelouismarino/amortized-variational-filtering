import torch
import numpy as np
import torchvision.transforms as torch_transforms
from PIL import Image

Compose = torch_transforms.Compose


class RandomRotation(object):
    """
    Rotates a PIL image or sequence of PIL images by a random amount.
    """
    def __init__(self, max_angle=15):
        self.max_angle = max_angle

    def __call__(self, input):
        angle = np.random.randint(-self.max_angle, self.max_angle)
        if type(input) == list:
            return [im.rotate(angle) for im in input]
        return input.rotate(angle)


class RandomCrop(object):
    """
    Randomly crops a PIL image or sequence of PIL images.
    """
    def __init__(self, output_size):
        if type(output_size) != tuple and type(output_size) != list:
            output_size = (output_size, output_size)
        self.output_size = output_size

    def __call__(self, input):
        img = input
        if type(input) == list:
            img = img[0]
        width, height = img.size[0], img.size[1]
        new_width, new_height = self.output_size
        left = np.random.randint(0, width - new_width)
        top = np.random.randint(0, height - new_height)
        if type(input) == list:
            return [im.crop((left, top, left + new_width, top + new_height)) for im in input]
        return input.crop((left, top, left + new_width, top + new_height))


class RandomHorizontalFlip(object):
    """
    Randomly flips a PIL image or sequence of PIL images horizontally.
    """
    def __init__(self):
        pass

    def __call__(self, input):
        flip = np.random.rand() > 0.5
        if flip:
            if type(input) == list:
                return [im.transpose(Image.FLIP_LEFT_RIGHT) for im in input]
            return input.transpose(Image.FLIP_LEFT_RIGHT)
        return input


class Resize(object):
    """
    Resizes a PIL image or sequence of PIL images.
    img_size can be an int, list or tuple (width, height)
    """
    def __init__(self, img_size):
        if type(img_size) != tuple and type(img_size) != list:
            img_size = (img_size, img_size)
        self.img_size = img_size

    def __call__(self, input):
        if type(input) == list:
            return [im.resize((self.img_size[0], self.img_size[1]), Image.BILINEAR) for im in input]
        return input.resize((self.img_size[0], self.img_size[1]), Image.BILINEAR)


class RandomSequenceCrop(object):
    """
    Randomly crops a sequence (list or tensor) to a specified length.
    """
    def __init__(self, seq_len):
        self.seq_len = seq_len

    def __call__(self, input):
        if type(input) == list:
            input_seq_len = len(input)
        elif 'shape' in dir(input):
            assert len(input.shape) > 1, 'Sequence does not have enough dimensions.'
            input_seq_len = input.shape[0]
        max_start_ind = input_seq_len - self.seq_len + 1
        assert max_start_ind > 0, 'Seqence length longer than input sequence.'
        start_ind = np.random.choice(range(max_start_ind))
        return input[start_ind:start_ind+self.seq_len]


class ConcatSequence(object):
    """
    Concatenates a sequence (list of tensors) along a new axis.
    """
    def __init__(self):
        pass

    def __call__(self, input):
        return torch.stack(input)


class ToTensor(object):
    """
    Converts a PIL image or sequence of PIL images into (a) PyTorch tensor(s).
    """
    def __init__(self):
        self.to_tensor = torch_transforms.ToTensor()

    def __call__(self, input):
        if type(input) == list:
            return [self.to_tensor(i) for i in input]
        return self.to_tensor(input)


class Normalize(object):
    """
    Normalizes a PyTorch tensor or a list of PyTorch tensors.
    """
    def __init__(self, mean, std):
        self.normalize = torch_transforms.Normalize(mean, std)

    def __call__(self, input):
        if type(input) == list:
            return [self.normalize(i) for i in input]
        return self.normalize(input)
