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
        self.hdul = fits.open(fits_file)
        self.image_data = self.hdul[0].data
        self.log_image_data = np.log(self.image_data)
        self.rows = self.image_data.shape[0]
        self.cols = self.image_data.shape[1]
    
    @staticmethod
    def open_image(image):
        """
        Opens the .fits image for viewing. This is for testing purposes to ensure that the image opened is correct. 
        The 'image' is really just a NumPy array. Note that in order to see some of the original 2D images,
        it must be plotted logarithmically.
        """
        plt.imshow(image, cmap="gray")
        plt.colorbar()
        plt.show()

    def open_self_image(self):
        plt.imshow(self.image_data, cmap="gray")
        plt.colorbar()
        plt.show()
        
    def get_dimensions(self):
        return (self.rows, self.cols)

    def overplot(self, spectrum):
        """
        Takes in a Spectrum object and plots it on top of the image.
        """
        #TODO: write this function
        pass
