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

    return np.array(intensity)


def plot_intensity(intensity_array):
    """
    Creates a plot of flux v. yvalue so that we can see how the yvalue flux changes at a particular x value.
    The maxima will be the locations of the orders. 
    """
    shifted_array = np.array(intensity_array) - 20 # this array is shifted down by 20. it will be used to detect
    # intersects where the intensity goes to 0

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

def find_step_boundaries(step_array, threshold=50):
    """
    Takes in an array that looks somewhat like a step function.
    Finds boundaries of the step function.
    """

def find_order_boundaries(intensity_array):
    """
    This function takes in a variable called `intensity_array` which contains a list of tuples 
    where each tuple contains the ypixel and the intensity of the pixel at a particular x value.
    Storing this x value will be important as we will be using these routines again to 
    do the same thing.
    """

    intensity_array = np.array(intensity_array)
    plt.show()


def main():
    fits_file = FitsFile("fits_files/r0760_stitched.fits")
    image = fits_file.image_data
    intensity_array = get_intensity_array(image)
    # plot_intensity(intensity_array) Plots the intensity array just to make sure that our routines are correct
    find_order_boundaries(intensity_array)

if __name__ == "__main__":
    main()
