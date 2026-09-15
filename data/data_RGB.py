import os
from .dataset_RGB import DataReader


def get_data(img_dir, inp, tar, mode='train', ori=False, img_options=None, text_detector=False):
    assert os.path.exists(img_dir)
    return DataReader(img_dir, inp, tar, mode, ori, img_options, text_detector=text_detector)
