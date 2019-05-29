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


def find_flux(image):
    xpixel = 1000
    intensity = []
    for row_num, row in enumerate(image):
        intensity.append((row_num, row[xpixel]))

    return intensity

def plot_intensity(intensity_array):
    """
    Creates a plot of flux v. yvalue so that we can see how the yvalue flux changes at a particular x value.
    The maxima will be the locations of the orders. 
    """
    yvalues = [item[0] for item in intensity_array]
    flux = [item[1] for item in intensity_array]
    plt.plot(yvalues, flux)
    plt.xlabel("ypixels")
    plt.ylabel("flux")
    plt.title("flux v. ypixel")
    plt.show()


def main():
    fits_file = FitsFile("fits_files/r0760_stitched.fits")
    image = fits_file.image_data
    plot_intensity(find_flux(image))

if __name__ == "__main__":
    main()
