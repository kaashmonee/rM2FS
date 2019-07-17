import numpy as np
import matplotlib.pyplot as plt
import scipy
from astropy import modeling
from gaussfit import Peak


class Spectrum:
    """
    This is a class that represents each white spot on the image.
    """
    spectrum_number = 1

    def __init__(self, xvalues, yvalues, fits_file):
        import util
        self.xvalues = xvalues
        self.yvalues = yvalues
        self.fits_file = fits_file

        # Remove very obvious outliers
        # self.xvalues, self.yvalues = util.sigma_clip(self.xvalues, self.yvalues, sample_size=30)

        self.true_yvals = None

        xlen = len(xvalues)
        ylen = len(yvalues)

        # Adding a correctness check to ensure that the dimensions of each are correct.
        if xlen != ylen:
            raise ValueError("The dimensions of the xvalues and yvalues array are not the same; xlen:", xlen, " ylen:", ylen)


        # Removes overlapping portions of the spectrum
        self.__remove_overlapping_spectrum() 

        # Generating peaks after the spectrum is cleaned and narrowed
        self.peaks = [Peak(x, y) for x, y in zip(self.xvalues, self.yvalues)]

        # Peaks for which a Gaussian was unable to be fit
        self.unfittable_peaks = []

        # In order to establish a width profile, each fitted Gaussian also 
        # contains a standard deviation/width value that is stored and fitted
        # a spline.
        self.peak_width_spline_function = None
        self.peak_width_spline_rms = None
        self.widths = None
        
        # Increment the spectrum number after creation of a spectrum.
        Spectrum.spectrum_number += 1



    def plot(self):
        """
        Takes in an optional parameter `show` that shows the plot as well.
        """
        size = 5
        if self.true_yvals is not None:
            scatter_plot = plt.scatter(self.xvalues, self.true_yvals, s=size)

        else:
            scatter_plot = plt.scatter(self.xvalues, self.yvalues, s=size)
        
        return scatter_plot


    def fit_spectrum(self, domain, degree):
        """
        This function fits a polynomial of degree `degree` and saves
        the output on the input domain. It then saves the RMS goodness of fit
        value.
        """
        import util
        yvals = self.true_yvals if self.true_yvals is not None else self.yvalues
        f = scipy.interpolate.UnivariateSpline(self.xvalues, yvals)

        self.output = f(domain)

        # Calculate the RMS goodness of fit and display to the user if the 
        # fit is very bad.
        spline_yvals = f(self.xvalues)
        self.rms_value = util.rms(spline_yvals, yvals)


    def plot_fit(self):
        fit_plot = plt.plot(self.output, linewidth=0.1)
        return fit_plot

    def remove_outliers(self):
        import util
        sample_sizes = [10, 100]
        self.xvalues, self.true_yvals = util.sigma_clip(
            self.xvalues, self.true_yvals, sample_size=sample_sizes
        )

        xvalue_set = set(self.xvalues)
        self.peaks = [peak for peak in self.peaks if peak.x in xvalue_set]

        # Safety check to ensure there are no duplicate x pixels.
        assert(len(self.peaks) == len(self.xvalues))

    
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
        


    
    def fit_peak_widths(self):
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


