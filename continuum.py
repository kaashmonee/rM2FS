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



# Pick an x pixel to plot flux
# Find intensity of y pixels
# Plot intensity
# Obtain local maxima
# local maxima is the y value at which the continuum exists.
# rinse and repeat for a few more spectra and we are good 


def get_intensity_array(image, xpixel=1000):
    """
    Returns a list of tuples with each tuple containing the y pixel 
    and the magnitude of the intensity at the given x pixel.
    """
    intensity = []
    for row_num, row in enumerate(image):
        intensity.append(row[xpixel])

    return intensity


def plot_intensity(intensity_array):
    """
    Creates a plot of flux v. yvalue so that we can see how the yvalue flux changes at a particular x value.
    The maxima will be the locations of the orders. 
    """
    plt.plot(intensity_array)
    plt.xlabel("ypixels")
    plt.ylabel("flux")
    plt.title("flux v. ypixel")
    plt.show()

def find_flux_max(yvalues, flux):
    """
    Finds the local maxima so that we can obtain the y values at which the flux is greatest.
    """
    pass

def find_order_boundaries(intensity_array):
    """
    This function takes in a variable called `intensity_array` which contains a list of tuples 
    where each tuple contains the ypixel and the intensity of the pixel at a particular x value.
    Storing this x value will be important as we will be using these routines again to 
    do the same thing.
    """
    lower_flux_limit = 20
    flatline = np.full(len(intensity_array), lower_flux_limit)
    plt.plot(flatline)
    plt.show()


def main():
    fits_file = FitsFile("fits_files/r0760_stitched.fits")
    image = fits_file.image_data
    intensity_array = get_intensity_array(image)
    plot_intensity(intensity_array)
    find_order_boundaries(intensity_array)

if __name__ == "__main__":
    main()
