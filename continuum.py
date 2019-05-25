from astropy.io import fits
import matplotlib

import matplotlib.pyplot as plt


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

    def open_image(self):
        """
        Opens the .fits image for viewing. This is for testing purposes to ensure that the image opened is correct. 
        """
        plt.imshow(self.image_data, cmap="gray")
        plt.colorbar()
        plt.show()

    def locate_regions_of_interest(self):
        """
        Identifies spectral regions of interest.
        Returns a list of numpy arrays that contain points the x, and y pixel that we want to fit a polynomial to.
        """

    def fit_fourth_order_legendre_polynomial(self):
        """
        Uses the output from the locate_regions_of_interest() function and fits a 4th order Legendre polynomial to each array of 
        x and y coordinates contained in the interest list. The idea to fit a 4th order polynomial was obtained from [cite paper here].
        """

    def try_drawing(self):
        """
        Temporary routine to determine if it's possible to draw on .fits files.
        """



def main():
    """
    Main routine for testing purposes..
    """
    test_file = "fits_files/b0759_stitched.fits"
    c = ContinuumCreator(test_file)
    c.open_image()


     
if __name__ == "__main__":
    main()
