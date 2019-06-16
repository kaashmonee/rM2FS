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
            # y0 = get_previous_peak(fits_file, peak, spec_ind)
            # y1 = peak.y
            # y2 = get_next_peak(fits_file, peak, spec_ind)
            # rng = get_range(y1-10, y1, y1+10, image_height)
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


def fit_gaussian(fits_file, rng, peak, spec_ind=0):
    fits_image = fits_file.image_data
    
    # At a particular x value, this obtains the y values in the image so that 
    # we can get a set of points for which we want to fit the Gaussian.
    ystart = rng[0] # rng[0]-200 if rng[0]-200 >= 0 else 0
    yend = rng[1] # rng[1]+100 if rng[1]+100 <= fits_image.shape[1] else fits_file.image[1]
    yrange = np.arange(ystart, yend)
    yrange = non_int_to_int(yrange)
    peak.x = int(peak.x)
    # print("yrange.dtype:", yrange.dtype, "peak.x dtype:", peak.x.dtype)
    intensity = fits_image[yrange, peak.x]
    # intensity = fits_image[peak.x, yrange]
    print("peak.y:", peak.y)
    print("intensity at peak:", fits_image[peak.x, peak.y])
    plt.plot(intensity)
    plt.show()
    print("fits_image.shape:", fits_image.shape)
    print("intensity:",intensity)

    # safety check to ensure same number of my points
    assert(len(intensity) == len(yrange))

    # Fits the intensity profile to an array of 
    mean, std = scipy.stats.norm.fit(intensity)
    m = modeling.models.Gaussian1D(mean=mean, stddev=std)

    output = m(yrange)
    print(output)

    peak.true_center = max(output)
            

def get_previous_peak(fits_file, peak, spec_ind):
    """
    This function obtains the previous peak in the same column in the image so that
    we can find the midpoint between the peaks to choose the y-domain for our 
    Gaussian fit.
    """

    # If this is the first spectrum, then the previous peak is the start of the 
    # picture.
    if spec_ind == 0:
        return 0

    cur_spectrum = fits_file.spectra[spec_ind]
    previous_spectrum = fits_file.spectra[spec_ind-1]
    prev_peak_ind = np.where(np.array(previous_spectrum.xvalues) == peak.x)
    ynminus1 = previous_spectrum.yvalues[prev_peak_ind[0][0]]

    return ynminus1

def get_next_peak(fits_file, peak, spec_ind):
    """
    This function obtains the next peak in the same column in the image so that
    we can find the midpoint between the currnet peak and the next peak to 
    choose a y-domain to fit a Gaussian.
    """

    # If this is the last spectrum, then the next peak is the end of the picture
    if spec_ind == len(fits_file.spectra) - 1:
        return fits_file.fits_image.shape[1]

    cur_spectrum = fits_file.spectra[spec_ind]
    next_spectrum = fits_file.spectra[spec_ind+1]
    next_peak_ind = np.where(np.array(next_spectrum.xvalues) == peak.x)
    yplus1 = next_spectrum.yvalues[next_peak_ind[0][0]]
    

    return yplus1

# fits file
# have spectra
# obtain the peaks and the points adjacent to the peaks
# generate distribution
# fit distribution
# plot distribution
