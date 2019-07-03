import numpy as np
import matplotlib.pyplot as plt
import scipy
from astropy import modeling
from gaussfit import Peak


class Spectrum:
    """
    This is a class that represents each white spot on the image.
    """
    def __init__(self, xvalues, yvalues):
        self.xvalues = xvalues
        self.yvalues = yvalues

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



    def plot(self):
        """
        Takes in an optional parameter `show` that shows the plot as well.
        """

        if self.true_yvals is not None:
            scatter_plot = plt.scatter(self.xvalues, self.true_yvals)

        else:
            scatter_plot = plt.scatter(self.xvalues, self.yvalues)
        
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
        fit_plot = plt.plot(self.output)
        return fit_plot

    def remove_outliers(self):
        import util
        self.xvalues, self.true_yvals = util.sigma_clip(self.xvalues, self.true_yvals)

    
    def __remove_overlapping_spectrum(self):
        """
        This routine removes the part of the spectrum on either side that is 
        responsible that overlaps. This way, we only use the middle spectrum for 
        our analysis.
        """

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

        # No part of the spectrum is overlapping, so there is no need to ensure
        # remove anything.
        if len(diff_ind_list) < 2:
            return

        # Starting and ending indices of the self.xvalues that we ought consider
        startx = diff_ind_list[0] + 1
        endx = diff_ind_list[1]

        self.xvalues = self.xvalues[startx:endx]
        self.yvalues = self.yvalues[startx:endx]


