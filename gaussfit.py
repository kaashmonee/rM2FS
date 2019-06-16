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
    This function obtains the true spectra
    """

    import pdb 
    image_height = fits_file.image_data.shape[1]
    # Generate the spectra if this already hasn't been done.
    if fits_file.spectra is None:
        fits_file.get_spectra()

    for spec_ind, spectrum in enumerate(fits_file.spectra):
        for peak in spectrum.peaks:
            y1 = peak.y
            y0 = y1-3
            y2 = y1+3
            if y0 <= 0: y0 = 0
            if y2 >= image_height: y2 = image_height
            rng = (y0, y2)
            fit_gaussian(fits_file, rng, peak, spec_ind=spec_ind)


def get_range(y0, y1, y2, image_height):
    if y0 == 0:
        return (y0, y2)

    elif y2 == image_height:
        return (y0, y2)
    
    else:
        return ((y2-y1)/2, (y1-y0)/2)

def non_int_to_int(iterable):
    return [int(x) for x in iterable]

def gauss(x, a, x0, sigma):
    return a*scipy.exp(-x*(x-x0)**2/(2*sigma**2))


def fit_gaussian(fits_file, rng, peak, spec_ind=0):
    fits_image = fits_file.image_data
    
    # At a particular x value, this obtains the y values in the image so that 
    # we can get a set of points for which we want to fit the Gaussian.
    ystart = rng[0] # rng[0]-200 if rng[0]-200 >= 0 else 0
    yend = rng[1] # rng[1]+100 if rng[1]+100 <= fits_image.shape[1] else fits_file.image[1]
    yrange = non_int_to_int(np.arange(ystart, yend))
    peak.x = int(peak.x)
    intensity = fits_image[yrange, peak.x]
    plt.scatter(yrange, intensity, color="red")
    

    # safety check to ensure same number of my points
    assert(len(intensity) == len(yrange))
    n = len(intensity)

    x = np.array(yrange); y = np.array(intensity)
    mean = sum(x*y)/n
    sigma = sum(y*(x-mean)**2)/n

    popt, pcov = scipy.optimize.curve_fit(gauss, x, y, p0=[1, mean, sigma])
    plt.plot(x, gauss(x, *popt), label="fit")

    # Fits the intensity profile to an array of 
    # mean, std = scipy.stats.norm.fit(intensity)
    # m = modeling.models.Gaussian1D(mean=mean, stddev=std)
    # fitter = modeling.fitting.LevMarLSQFitter()
    # model = modeling.models.Gaussian1D()
    # fitted_model = fitter(model, yrange, intensity)

    # plt.plot(yrange, fitted_model(yrange))
    plt.show()

    # peak.true_center = max(output)
            
# fits file
# have spectra
# obtain the peaks and the points adjacent to the peaks
# generate distribution
# fit distribution
# plot distribution
