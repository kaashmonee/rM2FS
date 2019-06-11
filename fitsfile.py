from astropy.io import fits
import matplotlib
import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import cv2
import time
import scipy
from spectrum import Spectrum

### global variable for button toggling purposes, modeled off the matplotlib
# documentation here: https://matplotlib.org/2.1.2/gallery/event_handling/keypress_demo.html.


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

    def plot_spectra(self, show=False):
        """
       Plots the spectra on top of the image.
        """

        # importing inside function to avoid circular dependency issues
        import cleanup 

        # Setting up plotting...
        fig = plt.figure()
        thresholded_im = cleanup.threshold_image(self.image_data)

        ax_im = fig.add_subplot(1, 1, 1)
        ax_plt = fig.add_subplot(1, 1, 1)

        ax_im.imshow(thresholded_im, origin="lower", cmap="gray")
        
        image_rows = self.image_data.shape[0]
        image_cols = self.image_data.shape[1]
        degree = 3

        spectrum_scatter_plots = []
        fit_plots = []

        for spectrum in self.spectra:

            spectrum_scatter = spectrum.plot(ax_plt)
            spectrum.fit_spectrum(np.arange(0, image_cols), degree)
            fit_plot = spectrum.plot_fit(ax_plt)

            spectrum_scatter_plots.append(spectrum_scatter)
            fit_plots.append(fit_plot)


        if show: 
            plt.xlabel("xpixel")
            plt.ylabel("ypixel") 
            plt.title("Image " + self.get_file_name() + " with Spectral Continuum Fits")
            plt.show()


    def __get_intensity_array(self, xpixel=1000):
        """
        At a particular xpixle, (default is 1000), this function returns
        an array of intensity for each pixel in y.
        """
        intensity = []
        for row_num, row in enumerate(self.image_data):
            intensity.append(row[xpixel])

        return np.array(intensity)

    
    def __find_peaks(self, intensity_array):
        """
        Find peaks in the intensity array. The peaks correspond to each order of the spectrograph.
        """
        # ignores peaks with intensities less than 100
        peaks = scipy.signal.find_peaks(intensity_array, height=100) 
        
        return peaks[0]


    def get_spectra(self):
        
        # This is the pixel we use to determine how many spectra there are
        prime_pixel = 1000 
        xpixels = np.arange(self.image_data.shape[1])
        print("xpixels:", xpixels)

        list_of_IA = [] # list of intensity arrays
        list_of_peaks = []  

        # Finds the number of peaks based on the reference prime x pixel
        num_peaks_prime = self.__find_peaks(self.__get_intensity_array(xpixel=prime_pixel))
        used_xpixels = []

        # Obtain the y v. intensity for each x pixel so 
        # we can find peaks for various values in the domain.
        for xpixel in xpixels:
            ia = self.__get_intensity_array(xpixel=xpixel)
            list_of_IA.append(ia)
            peaks = self.__find_peaks(ia)

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
            spectra.append(Spectrum(xvalues, yvalues, self.image_data))

        self.spectra = spectra


    def find_true_centers(self):
        """
        This function finds the true center for each peak in the spectrum.
        """
        # Each element in this list contains a list of points to 
        # fit to the gaussian
        for spectrum in self.spectra:

            gfitpoints = []
            for xvalue, ypeak in zip(self.xvalues, self.yvalues):
                fit_points = self.__get_gaussian_coords(xvalue, ypeak)
                gfitpoints.append(fit_points)

            # Fit a gaussian
            fitted_models = self.__fit_gaussian(self.xvalues, gfitpoints)
            self.centers = [max(fit) for fit in fitted_models]

        return self.centers


    def plot_true_centers(self):
        plt.scatter(self.xvalues, self.centers)

    def __fit_gaussian(self, xvalues, gfitpoints):
        """
        Fits a gaussian to each distribution of xvalues and points in each gfitpoints.
        Obtained from 
        https://stackoverflow.com/questions/44480137/how-can-i-fit-a-gaussian-curve-in-python. 
        """
        fitter = modeling.fitting.LevMarLSQFitter()
        model = modeling.models.Gaussian1D()
        fitted_models = [fitter(model, xvalues, points) for points in gfitpoints]

        return fitted_models


    def __get_gaussian_coords(xvalue, peak):
        """
        Finds the width of the peaks.
        """

        # do not include pixels dimmer than 100 to be the center of the gaussian
        intensity_threshold = 100 
        cur_y = peak # the current y pixel
        gypixels = [] # y pixels to include in the gaussian fit

        ## Finds all y pixels that are brigher than 100 in either direction.

        gypixels.append(cur_y)

        while self.image[xvalue, cur_y] > intensity_threshold:
            cur_y -= 1
            gypixels.append(cur_y)

        cur_y = peak

        while self.image[xvalue, cur_y] > intensity_threshold:
            cur_y += 1
            gypixels.append(cur_y)

        gxpixels = np.repeat(xvalue, len(gypixels))
        gypixels = np.array(gypixels)

        return xpixels, gypixels


    def get_file_name(self):
        """
        Returns the name of the file independent of the path.
        """
        return self.fits_file[self.fits_file.rfind("/")+1:]



def is_rectangular(l):
    """
    This is a correctness check function that is used in get_spectra() routine below.
    It ensures that the given array l is rectangular.
    """
    for i in l:
        if len(i) != len(l[0]):
            return False

    return True
