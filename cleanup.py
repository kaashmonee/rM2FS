from astropy.io import fits
import matplotlib
import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import cv2
import time
from fitsfile import FitsFile


def threshold_image(image):
    """
    Threshold's the image so that any values that are less than are set to zero and any values greater than 1000 are set to 1.
    Returns the thresholded image.
    """
    threshold_value = 5000
    thresholded_image = (image < threshold_value) * image
    return thresholded_image


def bright_spot_detector(image):
    """
    Identifies spectral regions of interest. Finds bright spots in the image. Could be useful for identifying and removing cosmic rays.
    """
    thresholded_image = threshold_image(image)
    thresholded_8bit = bytescale(thresholded_image)
    FitsFile.open_image(thresholded_8bit)
    plt.contour(thresholded_8bit, levels=np.logspace(-4.7, -3., 10), colors="white", alpha=0.5)
    im2, contours, hierarchy = cv2.findContours(thresholded_8bit, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(thresholded_8bit, contours, 1, (0, 255, 0), 3)
    FitsFile.open_image(thresholded_8bit)


def bytescale(image, cmin=None, cmax=None, high=255, low=0):
    """
    This function is a deprecated SciPy 16 bit image scaler. It is used for converting the 16 bit .fits files
    into 8 bit images that we can use to perform contour and edge detection functions on in OpenCV.
    Obtained from: https://stackoverflow.com/questions/25485886/how-to-convert-a-16-bit-to-an-8-bit-image-in-opencv.
    """
    if image.dtype == np.uint8:
        return image

    if high > 255:
        high = 255
    if low < 0:
        low = 0
    if high < low:
        raise ValueError("'high' should be greater than or equal to 'low'.")
    
    if cmin is None:
        cmin = image.min()
    if cmax is None:
        cmax = image.max()

    cscale = cmax - cmin
    if cscale == 0:
        cscale = 1
    
    scale = float(high - low) / cscale
    bytedata = (image - cmin) * scale + low
    return (bytedata.clip(low, high) + 0.5).astype(np.uint8)


def main():
    """
    Main routine for testing purposes..
    """
    t1 = time.time()
    test_file = "fits_files/r0760_stitched.fits"
    img = FitsFile(test_file)
    img.open_self_image()
    t2 = time.time()
    print("Time taken: ", t2 - t1)

if __name__ == "__main__":
    main()