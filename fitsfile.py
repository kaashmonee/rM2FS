from astropy.io import fits
import matplotlib
import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import cv2
import time

class FitsFile:
    def __init__(self, fits_file):
        """
        Creates a class that opens and contains .fits image attributes. It takes in a fits image path.
        """
        self.fits_file = fits_file
        self.hdul = fits.open(fits_file)
        self.image_data = self.hdul[0].data
        self.log_image_data = np.log(self.image_data)
        self.rows = self.image_data.shape[0]
        self.cols = self.image_data.shape[1]


    def get_dimensions(self):
        return (self.rows, self.cols)

    def plot_spectra(self, spectra, show=False):
        """
        Plots the spectra on top of the image.
        """

        # importing inside function to fix circular dependency issue
        import cleanup 

        image = self.image_data
        thresholded_im = cleanup.threshold_image(image)
        plt.imshow(thresholded_im, origin="lower", cmap="gray")
        image_rows = image.shape[0]
        image_cols = image.shape[1]
        degree = 3
        for spectrum in spectra:
            spectrum.plot()
            spectrum.fit_polynomial(np.arange(0, image_cols), degree)
            spectrum.plot_fit()

        if show: 
            plt.xlabel("xpixel")
            plt.ylabel("ypixel") 
            plt.title("Image " + self.get_file_name() + " with Spectral Continuum Fits")
            plt.show()


    def get_file_name(self):
        """
        Returns the name of the file independent of the path.
        """
        return self.fits_file[self.fits_file.rfind("/")+1:]
