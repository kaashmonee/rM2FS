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


class Spectrum:
    """
    This is a class that represents each white spot on the image.
    """
    def __init__(self, xvalues, yvalues):
        self.xvalues = xvalues
        self.yvalues = yvalues

    def plot(self, show=False):
        plt.scatter(xvalues, yvalues)
        if show: plt.show()



# Pick an x pixel to plot flux
# Find intensity of y pixels
# Plot intensity
# Obtain local maxima
# local maxima is the y value at which the continuum exists.
# rinse and repeat for a few more spectra and we are good 


def get_intensity_array(image, xpixel=1000):
    """
    Returns an array of ypixels and their intensity at a given xpixel
    """
    intensity = []
    for row_num, row in enumerate(image):
        intensity.append(row[xpixel])

    return np.array(intensity)


def plot_intensity(intensity_array, show=False):
    """
    Creates a plot of flux v. yvalue so that we can see how the yvalue flux changes at a particular x value.
    The maxima will be the locations of the orders. 
    """
    plt.plot(intensity_array)

    if show: 
        plt.xlabel("ypixels")
        plt.ylabel("flux")
        plt.title("flux v. ypixel")
        plt.show()

def find_flux_max(yvalues, flux):
    """
    Finds the local maxima so that we can obtain the y values at which the flux is greatest.
    """
    pass

def find_peaks(intensity_array, show=False):
    """
    Find peaks in the intensity array. The peaks correspond to each order of the spectrograph.
    """
    peaks = scipy.signal.find_peaks(intensity_array, height=100) # ignores peaks with intensities less than 100
    
    plt.plot(intensity_array)

    # Makes a scatter plot of the location of the peaks (peaks[0]) and
    # the intensity value of the peaks (intensity_array[peaks[0]])
    plt.scatter(peaks[0], intensity_array[peaks[0]])

    if show: 
        plt.xlabel("ypixel")
        plt.ylabel("intensity")
        plt.title("intensity v. ypixel with peaks shown")
        plt.show()

    return peaks[0]

def is_rectangular(l):
    for i in l:
        if len(i) != len(l[0]):
            return False

    return True

def create_spectra(image):
    # so far we have the y pixels of all the orders.
    # now we just need to repeat this process for a bunch of xpixels and then fit an order through all of them.
    xpixels = [600, 800, 1000, 1200, 400]
    list_of_IA = []
    list_of_peaks = []
    for xpixel in xpixels:
        ia = get_intensity_array(image, xpixel=xpixel)
        list_of_IA.append(ia)
        peaks = find_peaks(ia)
        list_of_peaks.append(peaks)

    # Each order is identified by the index of peak. 
    # Testing that we are detecting the same number of orders every time.
    assert(is_rectangular(list_of_peaks))


def main():
    fits_file = FitsFile("fits_files/r0760_stitched.fits")
    image = fits_file.image_data
    something(image)

if __name__ == "__main__":
    main()
