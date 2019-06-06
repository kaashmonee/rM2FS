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
from matplotlib.widgets import Button
import cleanup
import argparse



class Spectrum:
    """
    This is a class that represents each white spot on the image.
    """
    def __init__(self, xvalues, yvalues):
        self.xvalues = xvalues
        self.yvalues = yvalues
        self.is_narrowed = False

        xlen = len(xvalues)
        ylen = len(yvalues)

        # Adding a correctness check to ensure that the dimensions of each are correct.
        if xlen != ylen:
            raise ValueError("The dimensions of the xvalues and yvalues array are not the same; xlen:", xlen, " ylen:", ylen)

        # Narrow the spectrum immediately upon initialization.
        self.__narrow_spectrum()

    def plot(self, show=False):
        """
        Takes in an optional parameter `show` that shows the plot as well.
        """
        plt.scatter(self.xvalues, self.yvalues)
        if show: plt.show()

    def fit_polynomial(self, domain, degree):
        """
        This function fits a polynomial of degree `degree` and returns the output on 
        the input domain. 
        """
        # what kind of polynomial should be fit here?
        self.poly = np.polyfit(self.xvalues, self.yvalues, degree)
        f = self.__construct_function(self.poly) # returns the function to apply
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
            for i, c in enumerate(poly_list[::-1]):
                y += c * x**i

            return y

        return f


    def __narrow_spectrum(self):
        """
        This function narrows the spectrum down from a naive peak finding method
        to ensuring that the peaks are no more than 1 or 2 pixels away from each
        other.
        """

        # Do not narrow an already narrowed spectrum.
        if self.is_narrowed is True:
            return

        prev_y_pixel = self.yvalues[0]
        narrowed_y = []
        narrowed_x = []

        for ind, ypixel in enumerate(self.yvalues):
            if ypixel >= prev_y_pixel - 1 and ypixel <= prev_y_pixel + 1:
                narrowed_y.append(ypixel)
                narrowed_x.append(self.xvalues[ind])

            prev_y_pixel = ypixel

        self.xvalues = np.array(narrowed_x)
        self.yvalues = np.array(narrowed_y)

        self.is_narrowed = True

