import skimage as ski
from scipy import ndimage as ndi
import numpy as np
import pandas as pd


def apply_gaussian_blur_correction(img,obj_size):

    gaussian_blurred_img = ski.filters.gaussian(
        ski.util.img_as_float(img),
        sigma = 2.5 * obj_size,
        preserve_range=False
    )

    gauss_corrected_img = img / gaussian_blurred_img

    return gauss_corrected_img

def apply_clahe_contrast_correction(img):

    clahe_corrected_img = ski.exposure.equalize_adapthist(ski.util.img_as_uint(img/img.max()))

    return clahe_corrected_img
