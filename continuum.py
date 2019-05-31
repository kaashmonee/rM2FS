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
from numpy.polynomial.legendre import legfit
from numpy.polynomial.legendre import Legendre


class Spectrum:
    """
    This is a class that represents each white spot on the image.
    """
    def __init__(self, xvalues, yvalues):
        self.xvalues = xvalues
        self.yvalues = yvalues

        xlen = len(xvalues)
        ylen = len(yvalues)

        # Adding a correctness check to ensure that the dimensions of each are correct.
        if xlen != ylen:
            raise ValueError("The dimensions of the xvalues and yvalues array are not the same; xlen:", xlen, " ylen:", ylen)

    def plot(self, show=False):
        plt.scatter(self.xvalues, self.yvalues)
        if show: plt.show()

    def fit_polynomial(self, domain):
        # what kind of polynomial should be fit here?
        # fitting a 4th order legendre polynomial
        poly = np.polyfit(self.xvalues, self.yvalues, 4)
        f = self.__construct_function(poly)
        self.output = f(domain)

    def plot_fit(self, show=False):
        plt.plot(self.output)


    def __construct_function(self, poly_list):
        """
        Constructs a polynomial function based on the coefficients
        in the polynomial list and returns the function.
        """
        def f(x):
            y = np.zeros(len(x))
            for i, c in enumerate(poly_list):
                y += c * x**i

            return y

        return f


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


def find_peaks(intensity_array, show=False):
    """
    Find peaks in the intensity array. The peaks correspond to each order of the spectrograph.
    """
    peaks = scipy.signal.find_peaks(intensity_array, height=100) # ignores peaks with intensities less than 100
    

    if show: 
        plt.plot(intensity_array)

        # Makes a scatter plot of the location of the peaks (peaks[0]) and
        # the intensity value of the peaks (intensity_array[peaks[0]])
        plt.scatter(peaks[0], intensity_array[peaks[0]])
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

def get_spectra(image):
    # so far we have the y pixels of all the orders.
    # now we just need to repeat this process for a bunch of xpixels and then fit an order through all of them.
    xpixels = [600, 800, 1000, 1200, 400]
    list_of_IA = []
    list_of_peaks = []
    for xpixel in xpixels:
        ia = get_intensity_array(image, xpixel=xpixel)
        list_of_IA.append(ia)
        peaks = find_peaks(ia)
        list_of_peaks.append(np.array(peaks))

    # numpifying list_of_peaks
    list_of_peaks = np.array(list_of_peaks)

    # Each order is identified by the index of peak. 
    # Testing that we are detecting the same number of orders every time.
    assert(is_rectangular(list_of_peaks))

    lopt = np.transpose(np.array(list_of_peaks))
    spectra = []
    xvals = []
    for x in xpixels:
        xvals.append(np.repeat(x, len(list_of_peaks[0])))
    
    # numpifying array
    xvals = np.array(xvals)

    # Correctness check
    assert(np.array(xvals).shape == np.array(list_of_peaks).shape)

    num_cols = len(list_of_peaks[0])
    for i in range(num_cols):
        xvalues = xvals[:, i]
        yvalues = list_of_peaks[:, i]
        spectra.append(Spectrum(xvalues, yvalues))

    return spectra

def plot_spectra(image, spectra, show=False):
    plt.imshow(image)
    image_rows = image.shape[0]
    image_cols = image.shape[1]
    for spectrum in spectra:
        spectrum.plot()
        spectrum.fit_polynomial(np.arange(0, image.shape[1]))
        spectrum.plot_fit()

    if show: plt.show()


def main():
    fits_file = FitsFile("fits_files/r0760_stitched.fits")
    image = fits_file.image_data
    spectra = get_spectra(image)
    plot_spectra(image, spectra, show=True)



if __name__ == "__main__":
    main()

