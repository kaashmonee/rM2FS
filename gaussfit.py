# Routines for gauss fitting.
import numpy as np
import scipy

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
            y0 = get_previous_peak(fits_file, peak, spec_ind)
            y1 = peak.y
            y2 = get_next_peak(fits_file, peak, spec_ind)
            rng = get_range(y0, y1, y2, image_height)
            fit_gaussian(fits_file, rng, peak, spec_ind=spec_ind)


def get_range(y0, y1, y2, image_height):
    if y0 == 0:
        return (y0, y1)

    elif y2 == image_height:
        return (y1, y2)
    
    else:
        return ((y2-y1)/2, (y1-y0)/2)

def fit_gaussian(fits_file, rng, peak, spec_ind=0):
    fits_image = fits_file.image_data
    # print("spectral index:", spec_ind)
    # print("rng[0]", rng[0])
    # assert(isinstance(rng[0], np.int64))
    # assert(isinstance(rng[1], np.int64))
    # print("shape:", fits_image.shape)
    intensity = fits_image[peak.x, rng[0]:rng[1]]
    yrange = np.arange(rng[0], rng[1])

    # safety check to ensure same number of x and y points
    assert(len(intensity) == len(yrange))

    # Fits the intensity profile to an array of 
    mean, std = scipy.stats.norm.fit(intensity)
    m = modeling.models.Gaussian1D(mean=mean, stddev=std)
    output = m(yrange)

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

    import pdb

    # If this is the last spectrum, then the next peak is the end of the picture
    if spec_ind == len(fits_file.spectra) - 1:
        return fits_file.fits_image.shape[1]

    cur_spectrum = fits_file.spectra[spec_ind]
    next_spectrum = fits_file.spectra[spec_ind+1]
    next_peak_ind = np.where(np.array(next_spectrum.xvalues) == peak.x)
    yplus1 = next_spectrum.yvalues[next_peak_ind[0][0]]
    # pdb.set_trace()
    

    return yplus1






# fits file
# have spectra
# obtain the peaks and the points adjacent to the peaks
# generate distribution
# fit distribution
# plot distribution


