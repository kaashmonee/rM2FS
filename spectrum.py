import numpy as np
import matplotlib.pyplot as plt
import scipy
from astropy import modeling
import gaussfit
from gaussfit import Peak


class Spectrum:
    """
    This is a class that represents each white spot on the image.
    """
    spectrum_number = 1

    def __init__(self, xvalues, yvalues, fits_file):
        import util
        # These variables must be lists so that we can add spectra. They will
        # be turned into numpy arrays after we call the `build` function.
        self.xvalues = list(xvalues)
        self.yvalues = list(yvalues)
        self.fits_file = fits_file
        self.image_rows = self.fits_file.image_data[0]
        self.image_cols = self.fits_file.image_data[1]

        # Peaks for which a Gaussian was unable to be fit
        self.unfittable_peaks = []

        # In order to establish a width profile, each fitted Gaussian also 
        # contains a standard deviation/width value that is stored and fitted
        # a spline.
        self.peak_width_spline_function = None
        self.peak_width_spline_rms = None
        self.widths = None

        Spectrum.spectrum_number += 1

    
    def add_peak(self, x, y):
        """
        Adds a peak to the spectrum.
        """
        self.xvalues.append(x)
        self.yvalues.append(y)



    def build(self):
        import util
        """
        After all the points have been added to the spectrum, this function
        must be called to 'build' the spectrum, which performs Gaussian fits of 
        the integer peaks, removes outliers, removes the overlapping portions
        of the spectra, and establishes a width profile. If there are fewer than
        100 points in the spectrum, then the build is rejected and False
        is returned.
        """
        xlen = len(self.xvalues)
        ylen = len(self.yvalues)

        assert xlen == ylen
        if xlen <= 100:
            print("Build rejected! Fewer than 100 points in the spectrum...")
            return False

        # Sorting the x and y values
        self.xvalues, self.yvalues = util.sortxy(self.xvalues, self.yvalues)

        # Ensuring that we keep track of the integer yvalues
        # This is useful for when we want to plot the brightness vs. x value of
        # the peaks.
        self.int_xvalues = np.array(self.xvalues)
        self.int_yvalues = np.array(self.yvalues)

        if np.diff(self.xvalues).all() <= 0:
            print("self.xvalues:", self.xvalues)
            plt.plot(self.xvalues)
            plt.show()

        # Adding a correctness check to ensure that the dimensions of each are correct.
        if xlen != ylen:
            raise ValueError("The dimensions of the xvalues and yvalues array are not the same; xlen:", xlen, " ylen:", ylen)


        # Removes overlapping portions of the spectrum
        self.__remove_overlapping_spectrum() 

        # Generating peaks after the spectrum is cleaned and narrowed
        self.peaks = [Peak(x, y) for x, y in zip(self.xvalues, self.yvalues)]

        # Fits a Gaussian to each of the peaks.
        self.__fit_peak_gaussians()

        # Removes outliers
        self.__remove_outliers()

        # After obtaining the true y values and narrowing the spectrum,
        # we want to fit the spectrum with a UnivariateSpline, which is 
        # what this function does.
        degree = 3
        self.__fit_spectrum(np.arange(0, len(self.image_cols)), degree)

        # This function fits a spline to the peak widths and generates an rms 
        # value.
        self.__fit_peak_widths()

        return True


    def __fit_peak_gaussians(self):
        """
        This function fits a Gaussian to each spectrum in the .fitsfile. Then, 
        it finds the center of the Gaussian and assigns it to the peak.true_center
        parameter of each Peak object in each Spectrum. 
        """
        image = self.fits_file.image_data
        image_height = image.shape[1]

        print("Fitting gaussian...")
        import time
        t1 = time.time()
        success_counter = 0

        for peak in self.peaks:
            y1 = peak.y
            
            # This is just an arbitrary range that has been chosen.
            # This might have to be tweaked for various spectra.
            left_range = 4
            right_range = 4
            y0 = y1 - left_range
            y2 = y1 + right_range

            # Ensure that the ranges do not exceed the width and height of 
            # the image
            if y0 <= 0: y0 = 0
            if y2 >= image_height: y2 = image_height
            rng = (y0, y2)

            # This does the fitting and the peak.true_center setting.
            show = False
            success = gaussfit.fit_gaussian(self.fits_file, rng, peak,
                                            show=show)
            success_counter += success

        if success_counter != len(self.peaks):
            print("\nSpectrum %d had %d/%d points with a successful Gaussian fit." % (Spectrum.spectrum_number, success_counter, len(self.peaks)))


        self.yvalues = np.array([peak.true_center for peak in self.peaks])

        #############################################################
        # This is for testing purposes only --- this code should be #
        # deleted after testing is complete because no plotting sh- #
        # ould be taking place here.                                # 
        #############################################################

        # if Spectrum.spectrum_number in [15, 21]:
        #     self.fits_file.plot_spectra(num_to_plot=Spectrum.spectrum_number, 
        #                                 show=True, save=False) 
            # spectrum.plot_peak_widths()

        # if spec_ind == len(self.spectra):
        #     self.plot_spectra(num_to_plot=spec_ind, save=True)




    def plot(self, only_endpoints=True):
        """
        Takes in an optional parameter `show` that shows the plot as well.
        """
        size = 0.75

        xvalues_to_plot = self.xvalues
        yvalues_to_plot = self.yvalues

        if only_endpoints:
            xvalues_to_plot = [xvalues_to_plot[0], xvalues_to_plot[-1]]
            yvalues_to_plot = [yvalues_to_plot[0], yvalues_to_plot[-1]]

        scatter_plot = plt.scatter(xvalues_to_plot, yvalues_to_plot, s=size)

        
        return scatter_plot


    def __fit_spectrum(self, domain, degree):
        """
        This function fits a polynomial of degree `degree` and saves
        the output on the input domain. It then saves the RMS goodness of fit
        value.
        """
        import util
        self.output, self.rms_value = util.fit_spline(self.xvalues, self.yvalues,
                                                      domain, degree=degree)




    def plot_spectrum_brightness(self, num):
        """
        Plots the brightness of each spectra against the xvalue.
        """
        import util
        plt.clf()

        # Obtaining all the brightness values and plotting them against x
        brightness_array = self.fits_file.image_data[self.int_yvalues, self.int_xvalues]

        # Sigma clipping the brightness array to get rid of the extreme values
        self.int_xvalues, brightness_array = util.sigma_clip(self.int_xvalues, brightness_array, sample_size=100)
        plt.scatter(self.int_xvalues, brightness_array)
        
        # Smoothing the brightness array
        smoothed_brightness = scipy.signal.savgol_filter(brightness_array, 501, 5)

        # Obtaining the minima of the smoothed function and the x indices of the
        # minima
        extrema_indices = scipy.signal.argrelextrema(smoothed_brightness, np.less, order=100)
        extremax = self.int_xvalues[extrema_indices]

        # Plotting the minima
        extremabright = smoothed_brightness[extrema_indices]
        plt.scatter(extremax, extremabright)

        # Correctness check to ensure that the smoothed brightness array is the 
        # same length as the original array
        assert len(smoothed_brightness) == len(self.int_xvalues)

        # Plotting the smoothed brightness on top of everything        
        plt.plot(self.int_xvalues, smoothed_brightness, color="red")

        # Setting up plotting... 
        image_name = self.fits_file.get_file_name()
        plt.title("brightness vs. xvalues in %s, spectrum #: %d" % (image_name, num))
        plt.xlabel("xpixel")
        plt.ylabel("brightness")
        
        # plt.show()




    def plot_fit(self):
        linewidth = 0.25

        fit_plot = plt.plot(self.output, linewidth=linewidth)
        return fit_plot

    def __remove_outliers(self):
        import util
        sample_sizes = [10, 20, 50, 70, 100, 300]
        self.xvalues, self.yvalues = util.sigma_clip(
            self.xvalues, self.yvalues, sample_size=sample_sizes, sigma=3
        )

    
    def __remove_overlapping_spectrum(self):
        """
        This routine removes the part of the spectrum on either side that is 
        responsible that overlaps. This way, we only use the middle spectrum for 
        our analysis.
        """

        # Finds the differences between 2 adjacent elements in the array.
        diff_array = np.ediff1d(self.xvalues) 

        # Diff threshold to detect overlapping spectra
        diff_threshold = 30

        # Contains list of indices where next index differs by more than 
        # diff_threshold
        diff_ind_list = [] 

        for ind, diff in enumerate(diff_array):
            if diff >= diff_threshold:
                diff_ind_list.append(ind)


        # No part of the spectrum is overlapping, so there is no need to ensure
        # remove anything.
        if len(diff_ind_list) < 2:
            return

        # Finds the largest difference between indices that differ by 
        # diff_threshold pixels.
        diff_between_difs = np.ediff1d(diff_ind_list)
        max_diff_ind = np.argmax(diff_between_difs)

        # This is so that we can obtain the starting and ending index of the 
        # x values in the diff_ind_list[] list.
        startx_ind = max_diff_ind
        endx_ind = max_diff_ind + 1

        startx = diff_ind_list[startx_ind]
        endx = diff_ind_list[endx_ind]

        self.xvalues = self.xvalues[startx:endx]
        self.yvalues = self.yvalues[startx:endx]

        assert len(self.xvalues) == len(self.yvalues)
        


    
    def __fit_peak_widths(self):
        """
        For each peak, fits a spline to a plot to the function of the standard
        deviation fitted Gaussian to the x value.
        """
        import util
        # self.widths = np.array([peak.width for peak in self.peaks])
        self.widths = []
        xvalues = []

        # Ensuring that the peaks with unfittable Gaussians won't be included
        for peak in self.peaks:
            if peak.width == "failed":
                self.unfittable_peaks.append(peak)
                continue
            self.widths.append(peak.width)
            xvalues.append(peak.x)

        self.widths = np.array(self.widths)
        xvalues = np.array(xvalues)

        # TODO: This should be replaced with the more abstract util.fit_spline
        # function --- that is the function that should be used for all spline
        # fitting in this codebase. This is addressed in #93.
        f = scipy.interpolate.UnivariateSpline(xvalues, self.widths)
        widths_spline = f(xvalues)
        self.peak_width_spline_function = f
        self.peak_width_spline_rms = util.rms(widths_spline, self.widths)


    def plot_peak_widths(self):
        """
        Plotting function to plot the peak widths. This can only be called after
        gaussfit.fit_gaussian is called. It then fits a univariate spline to it. 
        The fit_peak_widths function must be called in order for this function
        to run.
        """
        xvalues = self.xvalues

        # Safety check to ensure that the user fits the peak widths before 
        # trying to plot them.
        if self.widths is None:
            raise RuntimeError("The plot_peak_widths function was called before the fit_peak_widths function was called.")

        widths = self.widths
        widths_spline = self.peak_width_spline_function(xvalues)

        plt.scatter(xvalues, widths)
        plt.plot(xvalues, widths_spline, color="red")
        plt.xlabel("xpixel")
        plt.ylabel("width")
        plt.title("gaussian width v. peak")
        
        # Adding the rms of the spline fit to the plot.
        ax = plt.gca()
        rms_text = "rms: " + str(self.peak_width_spline_rms)
        ax.text(5, 5, rms_text)

        plt.show()


