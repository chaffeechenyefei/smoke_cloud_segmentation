import numpy as np
import torch

def normal_imagenet(img_cv):
    """
    :param img_cv: [h,w,c] bgr 
    :return: 
    """
    mean = [123.675, 116.28, 103.53]
    std = [58.395, 57.12, 57.375]
    if isinstance(img_cv,list):
        return [ (im -mean)/std for im in img_cv ]
    else:
        return (img_cv-mean)/std

def hwc_to_chw(img_cv):
    if isinstance(img_cv,list):
        return [  np.transpose(im,axes=(2,0,1)) for im in img_cv ]
    else:
        return np.transpose(img_cv,axes=(2,0,1))

