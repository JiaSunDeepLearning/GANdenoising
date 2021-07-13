import tensorflow as tf
import tensorlayer as tl
import skimage
from tensorlayer.prepro import *
# from config import config, log_config
#
# img_path = config.TRAIN.img_path

import scipy
import numpy as np


def get_imgs_fn(file_name, path):
    """ Input an image path and name, return an image array """
    # return scipy.misc.imread(path + file_name).astype(np.float)
    return scipy.misc.imread(path + file_name, mode='RGB')


def crop_sub_imgs_fn(x, is_random=True):
    # x = crop(x, wrg=384, hrg=384, is_random=is_random)
    x = crop(x, wrg=224, hrg=224, is_random=is_random)
    x = x / (255. / 2.)
    x = x - 1.
    # x = (x - 0.5)*2
    return x


def downsample_fn(x):
    # We obtained the LR images by downsampling the HR images using bicubic kernel with downsampling factor r = 4.
    x = imresize(x, size=[224, 224], interp='bicubic', mode=None)
    # sigma = 15
    # x = x + sigma * np.random.randn(224, 224, 3)
    x = x / (255. / 2.)
    x = x - 1.
    # x = (x - 0.5)*2
    return x


def imnoise_fn(x):
    #
    #
    x = x + 1
    x = x * (225. / 2.)
    sigma = 50.
    x = x + sigma * np.random.randn(224, 224, 3)
    # x = x + sigma * np.random.randn(x.shape)
    x = x / (255. / 2.)
    x = x - 1.
    # x = (x - 0.5)*2
    return x


def imnoise_valid_fn(x, size_of_image):
    #
    #
    x = x + 1
    x = x * (225. / 2.)
    sigma = 50.
    np.random.seed(47)
    x = x + sigma * np.random.randn(size_of_image[0], size_of_image[1], 3)
    # x = x + sigma * np.random.randn(x.shape)
    x = x / (255. / 2.)
    x = x - 1.
    # x = (x - 0.5)*2
    return x


def psnr_me(img1, img2):
    psnr1_temp = skimage.measure.compare_psnr(im_true=img1, im_test=img2, data_range=255)
    return psnr1_temp


def ssim_me(img1, img2):
    ssim1_temp = skimage.measure.compare_ssim(X=img1, Y=img2, multichannel=True)
    return ssim1_temp
