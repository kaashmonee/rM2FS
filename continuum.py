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


##############################
##### End Spectrum Class #####
##############################


def get_intensity_array(image, xpixel=1000):
    """
    At a particular xpixle, (default is 1000), this function returns
    an array of intensity for each pixel in y.
    """
    intensity = []
    for row_num, row in enumerate(image):
        intensity.append(row[xpixel])

    return np.array(intensity)


def plot_intensity(intensity_array, show=False):
    """
    Creates a plot of flux v. yvalue so that we can see how the intensity changes.
    The maxima will be the locations of the spectra. 
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
    # ignores peaks with intensities less than 100
    peaks = scipy.signal.find_peaks(intensity_array, height=100) 
    

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
    """
    This is a correctness check function that is used in get_spectra() routine below.
    It ensures that the given array l is rectangular.
    """
    for i in l:
        if len(i) != len(l[0]):
            return False

    return True

def get_spectra(image):
    
    # This is the pixel we use to determine how many spectra there are
    prime_pixel = 1000 
    xpixels = np.arange(image.shape[1])
    print("xpixels:", xpixels)

    list_of_IA = [] # list of intensity arrays
    list_of_peaks = []  

    # Finds the number of peaks based on the reference prime x pixel
    num_peaks_prime = find_peaks(get_intensity_array(image, xpixel=prime_pixel))
    used_xpixels = []

    # Obtain the y v. intensity for each x pixel so 
    # we can find peaks for various values in the domain.
    for xpixel in xpixels:
        ia = get_intensity_array(image, xpixel=xpixel)
        list_of_IA.append(ia)
        peaks = find_peaks(ia)

        # This helps ensure that the same number of spectra are detected in each
        # xpixel as the reference xpixel, which is tentatively set to 1000
        if len(peaks) == len(num_peaks_prime):
            list_of_peaks.append(np.array(peaks))
            used_xpixels.append(xpixel)

    # numpifying list_of_peaks
    # This array contains peaks for each x value 
    # in x pixels.
    list_of_peaks = np.array(list_of_peaks)

    # Each order is identified by the index of peak. 
    # Have to ensure that list_of_peaks is rectangular because
    # we have to ensure that we are detecting the same number
    # of spectra for each x value. Otherwise, something will be 
    # discontinuous.
    assert(is_rectangular(list_of_peaks))

    spectra = []
    xvals = [] # An array that contains x values of each pixel.

    for x in used_xpixels:
        xvals.append(np.repeat(x, len(list_of_peaks[0])))
    
    # numpifying array
    xvals = np.array(xvals)

    # Correctness check
    assert(np.array(xvals).shape == np.array(list_of_peaks).shape)

    num_cols = len(list_of_peaks[0])

    # list_of_peaks contains the peaks for a singular x value, and 
    # xvalues contain the same x values for the length of each 
    # array in list_of_peaks. So this simply takes each column
    # from each array, which gives us the x and y coordinates
    # for each spectrum.
    for i in range(num_cols):
        xvalues = xvals[:, i]
        yvalues = list_of_peaks[:, i]
        spectra.append(Spectrum(xvalues, yvalues))

    return spectra


def plot_spectra(fits_image, spectra, show=False):
    image = fits_image.image_data
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
        plt.title("Image " + fits_image.get_file_name() + " with Spectral Continuum Fits")
        plt.show()



def export_spectra(file_name, spectra):
    """
    Exports the fit polynomials. This can be run only after 
    Spectrum.fit_polynomial is run.
    """
    polynomials = np.array([spectrum.poly for spectrum in spectra])
    np.savetxt(file_name, polynomials, delimiter=",")

def perform_fits(fits_file):
    image = fits_file.image_data
    spectra = get_spectra(image)
    plot_spectra(fits_file, spectra, show=True)



def main():
    # Doing brief cmd line parsing.
    parser = argparse.ArgumentParser(description="Calculate continuum fits.") 
    parser.add_argument("--export", help="--export <outputfile>")
    parser.add_argument("-l", 
       help="use this flag to loop through all fits files", action="store_true")
    args = parser.parse_args()

    # We need to write a function that will automatically perform these routines
    # so that we can determine for which functions this code does/does not work.
    # For each one, we should identify the assertion that failed and see what we
    # can change so that it does work.

    directory = "fits_files/"
    default_path = directory + "r0760_stitched.fits"

    if args.l is not False:
        for fits_path in os.listdir(directory):
            fits_file = FitsFile(directory+fits_path)
            perform_fits(fits_file)
    else:
        fits_file = FitsFile(default_path)
        perform_fits(fits_file)



    if args.export is not None:
        file_name = args.export
        export_spectra(file_name, spectra) # exports the spectrum to a txt.



if __name__ == "__main__":
    main()

