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
        self.spectra = []
        self.num_spectra = 0

        # self.__get_spectra()


    def get_dimensions(self):
        return (self.rows, self.cols)

    def plot_spectra(self, num_to_plot=None, save=False, show=False):
        """
        Plots the spectra on top of the image.
        """
        # Clear the figure.
        plt.clf()

        if not show and not save:
            raise ValueError("You must either choose to save or show the spectra.")

        if num_to_plot is None: 
            num_to_plot = len(self.spectra)

        # importing inside function to avoid circular dependency issues
        import util

        # Setting up plotting...
        vmin, vmax = util.get_vmin_vmax(self.image_data)
        plt.imshow(self.image_data, origin="lower", cmap="gray", 
                                    vmin=vmin, vmax=vmax)
        
        image_rows = self.image_data.shape[0]
        image_cols = self.image_data.shape[1]
        degree = 3

        spectrum_scatter_plots = []
        fit_plots = []

        for spectrum in self.spectra[:num_to_plot]:
            
            # Uncomment this section if the scatter plot portion of the spectrum
            # is desired.
            spectrum_scatter = spectrum.plot(only_endpoints=True)
            spectrum_scatter_plots.append(spectrum_scatter)

            fit_plot = spectrum.plot_fit()
            fit_plots.append(fit_plot)

        plt.xlabel("xpixel")
        plt.ylabel("ypixel") 
        plt.title("Image " + self.get_file_name() + " with Spectral Continuum Fits\nSpectra " + str(num_to_plot) + "/" + str(self.num_spectra))
        plt.xlim(0, self.get_dimensions()[1])
        plt.ylim(0, self.get_dimensions()[0])

        current_fig = plt.gcf()

        if save: 
            directory = "completed_images/"
            image_file_name = self.get_file_name() + "_fitted.svg"
            print("Saving " + image_file_name + " to disk...")
            current_fig.savefig(directory + image_file_name, dpi=1500)

        if show: 
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

    
    def __find_peaks(self):
        """
        Find peaks in the intensity array. The peaks correspond to each order of the spectrograph.
        """
        import util

        prime_pixel = 1000
        length = self.image_data.shape[1]
        xpixels = np.arange(length)

        # Finds number of peaks based on reference x pixel
        intensity_array = self.__get_intensity_array(xpixel=prime_pixel)
        num_peaks_prime = util.find_int_peaks(intensity_array)
        used_xpixels = []

        # The dictionary containing peaks at each x value
        xpeaks = dict()

        # Obtain a dictionary of x pixels and the peaks at each x pixel.
        for xpixel in xpixels:
            ia = self.__get_intensity_array(xpixel=xpixel)
            peaks = util.find_int_peaks(ia)
            xpeaks[xpixel] = peaks

        return xpeaks


    def __plot_peaks(self, xpeaks):
        print("Plotting peaks")
        img = np.zeros((self.image_data.shape[0], self.image_data.shape[1]))
        for x in xpeaks:
            img[xpeaks[x], x] = 1


        plt.imshow(img, origin="lower")
        current_figure = plt.gcf()
        plt.show()

        print("Saving peak plot...") 
        current_figure.savefig("assets/peak_plot.svg", dpi=2000)        


    def get_spectra(self):
        import util

        xpeaks = self.__find_peaks()

        image_length = self.image_data.shape[1]

        start_pixel = 1000
        yvalues_at_start = xpeaks[start_pixel]
        self.num_spectra = len(yvalues_at_start)
        xthreshold = 3
        ythreshold = 3
        cur_num_spectra = 0
        
        # Going from right to left
        for num, y in enumerate(yvalues_at_start):
            cur_y = y
            s = Spectrum([], [], self)
            
            cur_x = start_pixel
            for next_spec_x in range(start_pixel+1, image_length):

                check_y = xpeaks[next_spec_x]
                # Check for xpixels to see if there exists a y pixel that's less
                # than some value away.
                spec_indices = np.where(abs(cur_y-check_y) <= ythreshold)[0]

                if len(spec_indices) > 0:
                    next_ind = spec_indices[0]
                    nexty = check_y[next_ind]
                    s.add_peak(next_spec_x, nexty)
                    cur_x = next_spec_x
                    cur_y = nexty

                if abs(next_spec_x - cur_x) >= xthreshold:
                    break

            cur_x = start_pixel
            cur_y = y
            for prev_spec_x in range(start_pixel-1, 0, -1):

                check_y = xpeaks[prev_spec_x]
                spec_indices = np.where(abs(cur_y-check_y) <= ythreshold)[0]

                if len(spec_indices) > 0:
                    prev_ind = spec_indices[0]
                    prevy = check_y[prev_ind]
                    s.add_peak(prev_spec_x, prevy)
                    cur_x = prev_spec_x
                    cur_y = prevy

                if abs(prev_spec_x - cur_x) >= xthreshold:
                    break

            build_success = s.build()
            if build_success: 
                cur_num_spectra += 1
                print("Building spectrum %d/%d" % (cur_num_spectra, self.num_spectra))
                self.spectra.append(s)
                print("Min x:", s.xvalues[0], "\nMax x:", s.xvalues[-1])
            else:
                self.num_spectra -= 1


    def plot_spectra_brightness(self):
        """
        Plots the brightness of each spectra against the xvalue.
        """
        import util
        plt.clf()

        for spectrum in self.spectra:
            brightness_array = self.image_data[spectrum.int_yvalues, spectrum.int_xvalues]
            plt.scatter(spectrum.int_xvalues, brightness_array)
            
            
            smoothed_brightness = scipy.signal.savgol_filter(brightness_array, 201, 5)
            assert len(smoothed_brightness) == len(spectrum.int_xvalues)
            
            print("length brightness:", len(smoothed_brightness))
            plt.plot(smoothed_brightness, color="red")
            
            # domain = np.arange(spectrum.int_xvalues[0], spectrum.int_xvalues[-1])
            # output, rms = util.fit_spline(xpeaks, brightness_peaks, domain)
            # plt.plot(output, color="red")

            image_name = self.get_file_name()
            plt.title("brightness vs. xvalues in %s" % (image_name))
            plt.xlabel("xpixel")
            plt.ylabel("brightness")
            
            plt.show()

        

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
