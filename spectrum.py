import numpy as np
import matplotlib.pyplot as plt
import scipy
from astropy import modeling


class Spectrum:
    """
    This is a class that represents each white spot on the image.
    """
    def __init__(self, xvalues, yvalues, image):
        self.xvalues = xvalues
        self.yvalues = yvalues
        self.is_narrowed = False
        self.image = image

        xlen = len(xvalues)
        ylen = len(yvalues)

        # Adding a correctness check to ensure that the dimensions of each are correct.
        if xlen != ylen:
            raise ValueError("The dimensions of the xvalues and yvalues array are not the same; xlen:", xlen, " ylen:", ylen)

        # Narrow the spectrum immediately upon initialization.
        self.__narrow_spectrum()
        
        # Removes overlapping portions of the spectrum
        self.__remove_overlapping_spectrum() 

    def plot(self, ax, show=False):
        """
        Takes in an optional parameter `show` that shows the plot as well.
        """
        scatter_plot = ax.scatter(self.xvalues, self.yvalues)
        
        if show: plt.show()
        
        return scatter_plot


    def fit_spectrum(self, domain, degree):
        """
        This function fits a polynomial of degree `degree` and returns the output on 
        the input domain. 
        """
        # what kind of polynomial should be fit here?
        # self.poly = np.polyfit(self.xvalues, self.yvalues, degree)
        # f = self.__construct_function(self.poly) # returns the function to apply

        f = scipy.interpolate.interp1d(self.xvalues, self.yvalues, 
                                       fill_value="extrapolate")

        self.output = f(domain)


    def plot_fit(self, ax):
        fit_plot = ax.plot(self.output)
        return fit_plot

    




    # Deprecated after PR #36. This routine is useful for polynomial fitting,
    # but instead we are using the scipy spline interpolation routine.
    # It will be left here for documentation purposes for now, but should be
    # removed later if deemed unnecessary.
    """ 
    def __construct_function(self, poly_list):
        '''
        Constructs a polynomial function based on the coefficients
        in the polynomial list and returns the function.
        '''
        def f(x):
            y = np.zeros(len(x))
            for i, c in enumerate(poly_list[::-1]):
                y += c * x**i

            return y

        return f
    """


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


