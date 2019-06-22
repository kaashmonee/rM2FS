import numpy as np
import matplotlib.pyplot as plt
import scipy
from astropy import modeling
from gaussfit import Peak


class Spectrum:
    """
    This is a class that represents each white spot on the image.
    """
    def __init__(self, xvalues, yvalues, image):
        self.xvalues = xvalues
        self.yvalues = yvalues
        self.image = image

        self.has_true_peak_vals = False
        self.true_yvals = None

        xlen = len(xvalues)
        ylen = len(yvalues)

        # Adding a correctness check to ensure that the dimensions of each are correct.
        if xlen != ylen:
            raise ValueError("The dimensions of the xvalues and yvalues array are not the same; xlen:", xlen, " ylen:", ylen)

        # Narrow the spectrum immediately upon initialization.
        self.narrow_spectrum()
        
        # Removes overlapping portions of the spectrum
        self.__remove_overlapping_spectrum() 

        # Generating peaks after the spectrum is cleaned and narrowed
        self.peaks = [Peak(x, y) for x, y in zip(self.xvalues, self.yvalues)]



    def plot(self, ax, show=False):
        """
        Takes in an optional parameter `show` that shows the plot as well.
        """

        if self.has_true_peak_vals:
            scatter_plot = ax.scatter(self.xvalues, self.true_yvals)

        else:
            scatter_plot = ax.scatter(self.xvalues, self.yvalues)
        
        if show: plt.show()
        
        return scatter_plot


    def fit_spectrum(self, domain, degree):
        """
        This function fits a polynomial of degree `degree` and returns the 
        output on the input domain. 
        """
        yvals = self.true_yvals if self.has_true_peak_vals else self.yvalues
        f = scipy.interpolate.UnivariateSpline(self.xvalues, yvals)

        self.spectrum_fit_function = f

        self.output = f(domain)


    def plot_fit(self, ax):
        fit_plot = ax.plot(self.output)
        return fit_plot

    
    def narrow_spectrum(self):
        """
        This function narrows the spectrum down from a naive peak finding method
        to ensuring that the peaks are no more than 2 pixels away from each
        other.
        """

        yvals = self.yvalues
        if self.has_true_peak_vals:
            yvals = self.true_yvals

        prev_y_pixel = yvals[0]
        narrowed_y = []
        narrowed_x = []

        for ind, ypixel in enumerate(yvals):
            if ypixel >= prev_y_pixel - 1 and ypixel <= prev_y_pixel + 1:
                narrowed_y.append(ypixel)
                narrowed_x.append(self.xvalues[ind])

            prev_y_pixel = ypixel

        self.xvalues = np.array(narrowed_x)

        if self.has_true_peak_vals: 
            self.true_yvals = np.array(narrowed_y)
        else: 
            self.yvalues = np.array(narrowed_y)


    def __remove_overlapping_spectrum(self):

        # Finds the differences between 2 adjacent elements in the array.
        diff_array = np.ediff1d(self.xvalues) 

        # Diff threshold to detect overlapping spectra
        diff_threshold = 20

        # Contains list of indices where next index differs by more than 
        # diff_threshold
        diff_ind_list = [] 

        for ind, diff in enumerate(diff_array):
            if diff >= diff_threshold:
                diff_ind_list.append(ind)

        # Starting and ending indices of the self.xvalues that we ought consider
        startx = diff_ind_list[0] + 1
        endx = diff_ind_list[1]

        self.xvalues = self.xvalues[startx:endx]
        self.yvalues = self.yvalues[startx:endx]


