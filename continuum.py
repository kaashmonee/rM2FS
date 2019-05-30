from astropy.io import fits
import matplotlib
import numpy as np
import scipy.signal
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

def find_peaks(intensity_array):
    """
    Find peaks in the intensity array. The peaks correspond to each order of the spectrograph.
    """
    peaks = scipy.signal.find_peaks(intensity_array, height=100) # ignores peaks with intensities less than 100
    print("peaks:", peaks)
    
    plt.plot(intensity_array)

    # Makes a scatter plot of the location of the peaks (peaks[0]) and
    # the intensity value of the peaks (intensity_array[peaks[0]])
    plt.xlabel("ypixel")
    plt.ylabel("intensity")
    plt.title("intensity v. ypixel with peaks shown")
    plt.scatter(peaks[0], intensity_array[peaks[0]], color="red")
    plt.show()



def main():
    fits_file = FitsFile("fits_files/r0760_stitched.fits")
    image = fits_file.image_data
    intensity_array = get_intensity_array(image)
    # plot_intensity(intensity_array) Plots the intensity array just to make sure that our routines are correct
    find_peaks(intensity_array)

if __name__ == "__main__":
    main()
