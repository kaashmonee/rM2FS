from astropy.io import fits
import matplotlib
import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import cv2


"""
Steps:
1. Open .fits file to view in Python.
2. 
"""
def open_fits_aplpy():
    gc = aplpy.FITSFigure("fits_files/b0759_stitched.fits")


class ContinuumCreator:
    def __init__(self, fits_file):
        """
        Creates a class that opens and contains .fits image attributes. It takes in a fits image path.
        """
        self.hdul = fits.open(fits_file)
        self.image_data = self.hdul[0].data
        self.rows = self.image_data.shape[0]
        self.cols = self.image_data.shape[1]

    def open_image(self):
        """
        Opens the .fits image for viewing. This is for testing purposes to ensure that the image opened is correct. 
        The 'image' is really just a NumPy array.
        """
        plt.imshow(self.image_data, cmap="gray", norm=LogNorm())

        # print(self.image_data)
        plt.colorbar()
        plt.show()

    def locate_regions_of_interest(self):
        """
        Identifies spectral regions of interest.
        Returns a list of numpy arrays that contain points the x, and y pixel that we want to fit a polynomial to.
        """
        # plt.contour(self.image_data, levels=np.logspace(-4.7, -3., 10), colors="white", alpha=0.5)
        self.open_image()


    def fit_fourth_order_legendre_polynomial(self):
        """
        Uses the output from the locate_regions_of_interest() function and fits a 4th order Legendre polynomial to each array of 
        x and y coordinates contained in the interest list. The idea to fit a 4th order polynomial was obtained from [cite paper here].
        """

    def try_drawing(self):
        """
        Temporary routine to determine if it's possible to draw on .fits files.
        """
        # Image shape is 2056 x 2048.
        # Creating a random array that goes from x \in (0, 2056) and y \in (0, 2048)
        new_img = np.zeros((2056, 2048))
        for i in range(2048):
            for k in range(900, 1000):
                new_img[k, i] = 30000

        # plt.imshow(new_img, cmap="gray")
        plt.imshow(new_img + self.image_data, cmap="gray")
        plt.colorbar()
        plt.show()

    # Returns dimensions of the image.
    def get_dimensions(self):
        return (self.rows, self.cols)





def main():
    """
    Main routine for testing purposes..
    """
    test_file = "fits_files/r0760_stitched.fits"
    c = ContinuumCreator(test_file)
    # c.open_image()
    # c.try_drawing()
    c.locate_regions_of_interest()


     
if __name__ == "__main__":
    main()
