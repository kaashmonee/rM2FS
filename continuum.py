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


def main():
    """
    Main routine for testing purposes..
    """
    test_file = "fits_files/b0759_stitched.fits"
    c = ContinuumCreator(test_file)
    c.open_image()


     
if __name__ == "__main__":
    main()
