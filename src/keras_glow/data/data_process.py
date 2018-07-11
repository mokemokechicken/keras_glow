from copy import copy
from glob import glob

import numpy as np
from PIL import Image
from PIL.Image import BILINEAR

from keras_glow.config import Config


class DataProcess:
    def __init__(self, config: Config):
        self.config = config
        self._image_files = None
        self.files_in_epoch = None
        self.index_in_epoch = 0
        self.y_images = None  # type: list

    def shuffle_and_init_training_data(self):
        self.files_in_epoch = copy(self.image_files)
        np.random.shuffle(self.files_in_epoch)
        self.index_in_epoch = 0
        self.y_images = []

    def provide_next_data(self, size=None):
        bs = size or self.config.batch_size
        while len(self.y_images) < bs:
            self.load_into_images()
        y = np.concatenate(self.y_images[:bs], axis=0)  # (bs, ph, pw, 3)
        self.y_images = self.y_images[bs:]
        return y

    def open_image(self, image_file):
        im_size = (self.config.image_width, self.config.image_height)
        image = Image.open(image_file)  # type: Image.Image
        if image.size != im_size:
            image = image.resize(im_size, resample=BILINEAR)
        return image

    def load_image(self, image_file):
        """create two list of ndarray(1, ph, pw, 1) from image_file
        
        :param image_file: 
        :return: 
        """
        orig_img = self.open_image(image_file)
        self.index_in_epoch = (self.index_in_epoch + 1) % self.image_count
        y_data = self.image_to_array(orig_img)  # shape=(ih, iw, 3)
        return y_data

    def load_into_images(self):
        y_data = self.load_image(self.files_in_epoch[self.index_in_epoch])
        self.index_in_epoch = (self.index_in_epoch + 1) % self.image_count
        self.y_images.extend([np.expand_dims(y_data, axis=0)])  # (1, ph, pw, 3)

    @staticmethod
    def image_to_array(image: Image.Image):
        """
        
        :param image:
        :rtype: np.ndarray
        :return: 
        """
        return np.array(image, dtype="uint8") / 255.

    @property
    def image_files(self):
        if not self._image_files:
            self._image_files = list(glob("%s/**/*.jpg" % self.config.image_dir))
        return self._image_files

    @property
    def image_count(self):
        return len(self.image_files)

