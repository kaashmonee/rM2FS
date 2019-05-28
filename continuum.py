from astropy.io import fits
import matplotlib
import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import cv2
import time


class ContinuumCreator:
    def __init__(self, fits_file):
        """
        Creates a class that opens and contains .fits image attributes. It takes in a fits image path.
        """
        self.hdul = fits.open(fits_file)
        self.image_data = self.hdul[0].data
        self.log_image_data = np.log(self.image_data)
        self.rows = self.image_data.shape[0]
        self.cols = self.image_data.shape[1]

    def open_image(self, image):
        """
        Opens the .fits image for viewing. This is for testing purposes to ensure that the image opened is correct. 
        The 'image' is really just a NumPy array. Note that in order to see some of the original 2D images,
        it must be plotted logarithmically.
        """
        plt.imshow(image, cmap="gray")
        plt.colorbar()
        plt.show()



    def fit_fourth_order_legendre_polynomial(self):
        """
        Uses the output from the locate_regions_of_interest() function and fits a 4th order Legendre polynomial to each array of 
        x and y coordinates contained in the interest list. The idea to fit a 4th order polynomial was obtained from [cite paper here].
        """


    # Returns dimensions of the image.
    def get_dimensions(self):
        return (self.rows, self.cols)


    ### These routines here could be useful for cosmic ray detection.    
    def threshold_image(self):
        """
        Threshold's the image so that any values that are less than are set to zero and any values greater than 1000 are set to 1.
        Returns the thresholded image.
        """
        self.threshold_value = 5000
        thresholded_image = (self.image_data < self.threshold_value) * self.image_data
        return thresholded_image

    
    def bright_spot_detector(self):
        """
        Identifies spectral regions of interest. Finds bright spots in the image. Could be useful for identifying and removing cosmic rays.
        """
        thresholded_image = self.threshold_image()
        thresholded_8bit = self.bytescale(thresholded_image)
        self.open_image(thresholded_8bit)
        plt.contour(thresholded_8bit, levels=np.logspace(-4.7, -3., 10), colors="white", alpha=0.5)
        im2, contours, hierarchy = cv2.findContours(thresholded_8bit, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(thresholded_8bit, contours, 1, (0, 255, 0), 3)
        self.open_image(thresholded_8bit)


    def bytescale(self, image, cmin=None, cmax=None, high=255, low=0):
        """
        This function is a deprecated SciPy 16 bit image scaler. It is used for converting the 16 bit .fits files
        into 8 bit images that we can use to perform contour and edge detection functions on in OpenCV.
        Obtained from: https://stackoverflow.com/questions/25485886/how-to-convert-a-16-bit-to-an-8-bit-image-in-opencv.
        """
        if image.dtype == np.uint8:
            return image

        if high > 255:
            high = 255
        if low < 0:
            low = 0
        if high < low:
            raise ValueError("'high' should be greater than or equal to 'low'.")
        
        if cmin is None:
            cmin = image.min()
        if cmax is None:
            cmax = image.max()

        cscale = cmax - cmin
        if cscale == 0:
            cscale = 1
        
        scale = float(high - low) / cscale
        bytedata = (image - cmin) * scale + low
        return (bytedata.clip(low, high) + 0.5).astype(np.uint8)





def main():
    """
    Main routine for testing purposes..
    """
    t1 = time.time()
    test_file = "fits_files/r0760_stitched.fits"
    c = ContinuumCreator(test_file)
    c.locate_regions_of_interest()
    t2 = time.time()
    print("Time taken: ", t2 - t1)


# def try_drawing(self):
#     """
#     Temporary routine to determine if it's possible to draw on .fits files.
#     """
#     # Image shape is 2056 x 2048.
#     # Creating a random array that goes from x \in (0, 2056) and y \in (0, 2048)
#     new_img = np.zeros((2056, 2048))
#     for i in range(2048):
#         for k in range(900, 1000):
#             new_img[k, i] = 30000

#     # plt.imshow(new_img, cmap="gray")
#     plt.imshow(new_img + self.image_data, cmap="gray")
#     plt.colorbar()
#     plt.show()
     
if __name__ == "__main__":
    main()
