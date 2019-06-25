# Routines for gauss fitting.
import numpy as np
import scipy
from astropy import modeling
import matplotlib.pyplot as plt
import math

class Peak:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.true_center = None


def non_int_to_int(iterable):
    return [int(x) for x in iterable]

def gauss(x, amp, cen, wid):
    """
    Gauss fitting function. Using the precision (tau) to define the width of the
    distribution, as mentioned here: 
    https://en.wikipedia.org/wiki/Normal_distribution.
    """
    return (amp / ((2*math.pi)**0.5 * wid)) * scipy.exp(-(x-cen)**2 / (2*wid**2))


def fit_gaussian(fits_file, rng, peak, show=False):
    """
    This function obtains the fitting parameters for each Gaussian profile. 
    This includes the mean, expected max, and the standard deviation. It then 
    uses those parameters on a "continuous" domain to obtain a nice looking 
    Gaussian from which we can obtain each peak.
    """
    fits_image = fits_file.image_data
    
    # At a particular x value, this obtains the y values in the image so that 
    # we can get a set of points for which we want to fit the Gaussian.
    ystart = rng[0]     
    yend = rng[1]

    # Creates the the range of yvalues for which we want to fit our Gaussian    
    yrange = np.arange(ystart, yend)

    # Grabs the intensity at each y value and the given x value
    intensity = fits_image[yrange, peak.x]
    
    # safety check to ensure same number of my points
    assert(len(intensity) == len(yrange))

    # x, y points for which we want to fit our Gaussian
    x = np.array(yrange)
    y = np.array(intensity)

    # We use a continuous domain as suggested here so that we have a 
    # smooth Gaussian:
    # https://stackoverflow.com/questions/42026554/fitting-a-better-gaussian-to-data-points
    x_continuous = np.linspace(yrange[0], yrange[-1], 100)

    # These are parameters used to construct the Gaussian.
    # We divide by sum(y) for the reason cited here:
    # https://stackoverflow.com/questions/44398770/python-gaussian-curve-fitting-gives-straight-line-supplied-amplitude-of-y
    peak_value = y.max()
    mean = sum(x*y)/sum(y)
    sigma = sum(y*(x-mean)**2)/sum(y)

    # To determine the p0 values, we used the information here:
    # https://stackoverflow.com/questions/29599227/fitting-a-gaussian-getting-a-straight-line-python-2-7.
    popt, pcov = scipy.optimize.curve_fit(gauss, x, y, 
                                          p0=[peak_value, mean, sigma], 
                                          maxfev=10000)


    mean_intensity = popt[0]
    mean_y = popt[1]
    peak.true_center = mean_y

    if show:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(yrange, intensity, color="red")
        ax.plot(x_continuous, gauss(x_continuous, *popt), label="fit")

        ax.annotate(
            "gaussian peak:(" + str(mean_y) + "," + str(mean_intensity) + ")", 
            xy=(mean_y, mean_intensity), 
            xytext=(mean_y+1, mean_intensity+1.5), 
            arrowprops=dict(facecolor="black", shrink=0.5)
        )

        plt.xlabel("ypixel")
        plt.ylabel("intensity")
        plt.title("Intensity v. ypixel for " + fits_file.get_file_name() + " at x=" + str(peak.x))
        plt.show()

