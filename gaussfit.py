# Routines for gauss fitting.
import numpy as np
import scipy
from astropy import modeling
import matplotlib.pyplot as plt

class Peak:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.true_center = None


def get_true_peaks(fits_file):
    """
    This function fits a Gaussian to each spectrum in the .fitsfile. Then, 
    it finds the center of the Gaussian and assigns it to the peak.true_center
    parameter of each Peak object in each Spectrum. 
    """

    import pdb 
    image_height = fits_file.image_data.shape[1]
    # Generate the spectra if this already hasn't been done.
    if fits_file.spectra is None:
        fits_file.get_spectra()

    for spec_ind, spectrum in enumerate(fits_file.spectra):
        for peak in spectrum.peaks:
            y1 = peak.y
            left_range = 5
            right_range = 6
            y0 = y1-left_range
            y2 = y1+right_range

            # Ensure that the ranges do not exceed the width and height of the 
            # image
            if y0 <= 0: y0 = 0
            if y2 >= image_height: y2 = image_height
            rng = (y0, y2)

            # This does the fitting and the peak.true_center setting.
            fit_gaussian(fits_file, rng, peak)



def non_int_to_int(iterable):
    return [int(x) for x in iterable]

def gauss(x, a, x0, sigma):
    """
    This is the Gaussian function. This takes the following parameters:
    x: xvalues
    a: yvalues
    x0: ?
    sigma: standard deviation
    """
    return a*scipy.exp(-x*(x-x0)**2/(2*sigma**2))


def fit_gaussian(fits_file, rng, peak):
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
    plt.scatter(yrange, intensity, color="red")
    

    # safety check to ensure same number of my points
    assert(len(intensity) == len(yrange))

    # x, y points for which we want to fit our Gaussian
    x = np.array(yrange)
    y = np.array(intensity)

    # We use a continuous domain as cited here so that we have a smooth Gaussian
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
                                      p0=[peak_value, mean, sigma], maxfev=5000)

    plt.plot(x_continuous, gauss(x_continuous, *popt), label="fit")

    plt.show()

