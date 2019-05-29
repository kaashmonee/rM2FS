from astropy.io import fits
import matplotlib
import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import cv2
import time
from fitsfile import FitsFile








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
