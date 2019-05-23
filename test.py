from astropy.io import fits
import matplotlib

import matplotlib.pyplot as plt


"""
Steps:
1. Open .fits file to view in Python.
"""
def open_fits():
    hdul = fits.open("fits_files/b0759_stitched.fits")
    print(hdul)
    print(hdul.info())
    image_data = hdul[0].data
    print(image_data.shape)
    print(type(image_data))
    plt.imshow(image_data, cmap="gray")
    plt.colorbar()
    print("Got here!")
    plt.show()

open_fits()
