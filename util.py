import astropy.stats
from astropy.io import fits
from astropy.io.misc import fnpickle
from astropy.io.misc import fnunpickle
import gaussfit
import numpy as np
import scipy.signal
import matplotlib.pyplot as plt
from fitsfile import FitsFile
from spectrum import Spectrum
from collections import Iterable


# Using built in Python serializers as adding some custom functionality to save 
# and load fits_files.
def save(fits_file):
    save_directory = "fitted_files/"
    fnpickle(fits_file, save_directory + fits_file.get_file_name() + ".pkl")

def load(fits_file_path):
    fits_file = fnunpickle(fits_file_path)
    return fits_file

def sigma_clip(xvalues, yvalues, sample_size=10, sigma=3):
    """
    Returns a 3 sigma clipped dataset that will perform sigma clipping on 10 
    adjacent x and y values.
    """


    def clip_helper(xvalues, yvalues, sample_size, sigma):

        length = len(xvalues)
        
        # Correctness check
        assert(len(xvalues) == len(yvalues))

        new_xvals = []
        new_yvals = []

        for i in range(0, length, sample_size):
            domain = xvalues[i:i+sample_size]
            data = yvalues[i:i+sample_size]

            # Performs a 3sigma clipping on every 10 pixels.
            output = astropy.stats.sigma_clip(data, sigma=sigma)

            new_xvals.extend(domain[~output.mask])
            new_yvals.extend(data[~output.mask])

        new_len = len(new_xvals)

        if (length-new_len) / length >= 0.1:
            print_warning("Over 10% of pixels have been rejected in the sigma_clip routine.")

        assert len(new_xvals) == len(new_yvals)

        return np.array(new_xvals), np.array(new_yvals)


    new_x, new_y = xvalues, yvalues

    if isinstance(sample_size, Iterable):
        for sample in sorted(sample_size, reverse=True):
            new_x, new_y = clip_helper(new_x, new_y, sample, sigma)
        
        return new_x, new_y

    else:
        return clip_helper(xvalues, yvalues, sample_size, sigma)





def rms(peaks, fitted_peaks):
    """
    Calculates the RMS given the peaks and the fitted peaks. 
    """
    return np.sqrt(np.mean(np.square(peaks - fitted_peaks)))


def print_warning(message):
    warning_string = """
    ==========================
    Warning: %s
    ==========================
    """
    print(warning_string % message)


def export_spectra(file_name, spectra):
    """
    Exports the fit polynomials. This can be run only after 
    Spectrum.fit_polynomial is run.
    """
    polynomials = np.array([spectrum.poly for spectrum in spectra])
    np.savetxt(file_name, polynomials, delimiter=",")

def perform_fits(fits_file):
    # Check if this file exists in the fitted_files/ directory
    fits_file.get_true_peaks()
    fits_file.plot_spectra(save=True)
    print("Saving %s to disk..." % (fits_file.get_file_name() + ".pkl"))
    save(fits_file)

def display_centers(fits_file):
    """
    This should display the gaussian and the fit for each peak.
    """
    fits_file.get_spectra()
    fits_file.find_true_centers()
    xvalues = fits_file.xdomain
    gfitpoints = fits_file.get_points_to_fit() 
    fitted_models = fits_file.get_fitted_models()

    plt.scatter(xvalues, gfitpoints[0])
    plt.plot(fitted_models[0])
    plt.show()

def threshold_image(image):
    """
    Threshold's the image so that any values that are less than are set to zero and any values greater than 1000 are set to 1.
    Returns the thresholded image.
    """
    threshold_value = 5000
    thresholded_image = (image < threshold_value) * image
    return thresholded_image


def detect_bright_spots(image):
    """
    Identifies spectral regions of interest. Finds bright spots in the image. Could be useful for identifying and removing cosmic rays.
    """
    thresholded_image = threshold_image(image)
    thresholded_8bit = bytescale(thresholded_image)
    FitsFile.open_image(thresholded_8bit)
    plt.contour(thresholded_8bit, levels=np.logspace(-4.7, -3., 10), colors="white", alpha=0.5)
    im2, contours, hierarchy = cv2.findContours(thresholded_8bit, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(thresholded_8bit, contours, 1, (0, 255, 0), 3)
    FitsFile.open_image(thresholded_8bit)


def bytescale(image, cmin=None, cmax=None, high=255, low=0):
    """
    This function is a deprecated SciPy 16 bit image scaler. It is used for converting the 16 bit .fits files
    into 8 bit images that we can use to perform contour and edge detection functions on in OpenCV.
    Obtained from: https://stackoverflow.com/questions/25485886/how-to-convert-a-16-bit-to-an-8-bit-image-in-opencv.
    """
    if image.dtype == np.uint8:
        return image

    if high > 255:
        high = 255
    if low < 0:
        low = 0
    if high < low:
        raise ValueError("'high' should be greater than or equal to 'low'.")
    
    if cmin is None:
        cmin = image.min()
    if cmax is None:
        cmax = image.max()

    cscale = cmax - cmin
    if cscale == 0:
        cscale = 1
    
    scale = float(high - low) / cscale
    bytedata = (image - cmin) * scale + low
    return (bytedata.clip(low, high) + 0.5).astype(np.uint8)

