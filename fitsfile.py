from astropy.io import fits
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import scipy
from spectrum import Spectrum
import gaussfit
import sys

### global variable for button toggling purposes, modeled off the matplotlib
# documentation here: https://matplotlib.org/2.1.2/gallery/event_handling/keypress_demo.html.


class FitsFile:
    def __init__(self, fits_file):
        """
        Creates a class that opens and contains .fits image attributes. It takes in a fits image path.
        """
        self.fits_file = fits_file
        hdul = fits.open(fits_file)
        self.image_data = hdul[0].data
        self.log_image_data = np.log(self.image_data)
        self.rows = self.image_data.shape[0]
        self.cols = self.image_data.shape[1]

        self.__get_spectra()


    def get_dimensions(self):
        return (self.rows, self.cols)

    def plot_spectra(self, num_to_plot=None, save=False, show=True):
        """
        Plots the spectra on top of the image.
        """

        if not show and not save:
            raise ValueError("You must either choose to save or show the spectra.")

        if num_to_plot is None: 
            num_to_plot = len(self.spectra)

        # importing inside function to avoid circular dependency issues
        import util

        # Setting up plotting...
        thresholded_im = util.threshold_image(self.image_data)

        plt.imshow(thresholded_im, origin="lower", cmap="gray")
        
        image_rows = self.image_data.shape[0]
        image_cols = self.image_data.shape[1]
        degree = 3

        spectrum_scatter_plots = []
        fit_plots = []

        print("Plotting %i of %i spectra" % (num_to_plot, len(self.spectra)))

        for spectrum in self.spectra[:num_to_plot]:

            spectrum_scatter = spectrum.plot()
            fit_plot = spectrum.plot_fit()

            spectrum_scatter_plots.append(spectrum_scatter)
            fit_plots.append(fit_plot)

        plt.xlabel("xpixel")
        plt.ylabel("ypixel") 
        plt.title("Image " + self.get_file_name() + " with Spectral Continuum Fits\nSpectra " + str(num_to_plot) + "/" + str(len(self.spectra)))
        plt.xlim(0, self.get_dimensions()[1])
        plt.ylim(0, self.get_dimensions()[0])

        current_fig = plt.gcf()
        
        if show: 
            plt.show()

        if save: 
            directory = "completed_images/"
            image_file_name = self.get_file_name() + "_fitted.png"
            print("Saving " + image_file_name + " to disk...")
            current_fig.savefig(directory + image_file_name)


    def get_true_peaks(self, show=False):
        """
        This function fits a Gaussian to each spectrum in the .fitsfile. Then, 
        it finds the center of the Gaussian and assigns it to the peak.true_center
        parameter of each Peak object in each Spectrum. 
        """

        image_height = self.image_data.shape[1]

        print("Fitting gaussian...")
        import time
        t1 = time.time()
        for spec_ind, spectrum in enumerate(self.spectra):
            
            sys.stdout.write("\rFitting spectrum %i/%i" % (spec_ind, len(self.spectra)))
            sys.stdout.flush()
            success_counter = 0

            for peak in spectrum.peaks:
                y1 = peak.y
                
                # This is just an arbitrary range that has been chosen.
                # This might have to be tweaked for various spectra.
                left_range = 4
                right_range = 4
                y0 = y1-left_range
                y2 = y1+right_range

                # Ensure that the ranges do not exceed the width and height of 
                # the image
                if y0 <= 0: y0 = 0
                if y2 >= image_height: y2 = image_height
                rng = (y0, y2)

                # This does the fitting and the peak.true_center setting.
                show = False
                success = gaussfit.fit_gaussian(self, rng, peak, show=show, spec_ind=spec_ind)
                success_counter += success

            if success_counter != len(spectrum.peaks):
                print("\nSpectrum %d had %d/%d points with a successful Gaussian fit." % (spec_ind, success_counter, len(spectrum.peaks)))


            spectrum.true_yvals = np.array([peak.true_center for peak in spectrum.peaks])
            spectrum.remove_outliers()

            # After obtaining the true y values and narrowing the spectrum,
            # we want to fit the spectrum with a UnivariateSpline, which is 
            # what this function does.
            degree = 3
            spectrum.fit_spectrum(np.arange(0, self.cols), degree)
            spectrum.fit_peak_widths()

            # if spec_ind == 20:
            #     t2 = time.time()
            #     print("time taken:", t2-t1)
            #     self.plot_spectra(num_to_plot=spec_ind, show=False, save=True) 
            #     spectrum.plot_peak_widths()

            # if spec_ind == 21:
            #     # import pdb; pdb.set_trace()
            #     t2 = time.time()
            #     print("time taken:", t2-t1)
            #     self.plot_spectra(show=True, num_to_plot=spec_ind) 


            # if spec_ind == 60:
            #     t2 = time.time()
            #     print("time taken:", t2-t1)
            #     self.plot_spectra(show=True, num_to_plot=spec_ind) 


            if spec_ind == len(self.spectra):
                self.plot_spectra(num_to_plot=spec_ind, save=True, show=False)





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


    def __get_spectra(self):
        
        # This is the pixel we use to determine how many spectra there are
        prime_pixel = 1000 
        xpixels = np.arange(self.image_data.shape[1])

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
            spectra.append(Spectrum(xvalues, yvalues, self))

        self.spectra = spectra


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
