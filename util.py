import astropy.stats
from astropy.io import fits
from astropy.io.misc import fnpickle
from astropy.io.misc import fnunpickle
import gaussfit
import numpy as np
import scipy.signal
import matplotlib.pyplot as plt
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
    adjacent x and y values. This does not modify the original xvalues and 
    yvalues arrays.
    """


    def clip_helper(xvalues, yvalues, sample_size, sigma):

        length = len(xvalues)
        
        # Correctness check
        assert(len(xvalues) == len(yvalues))

        new_xvals = []
        new_yvals = []

        for i in range(0, length, sample_size):
            domain = np.array(xvalues[i:i+sample_size])
            data = np.array(yvalues[i:i+sample_size])

            # Performs a 3sigma clipping on every 10 pixels.
            output = astropy.stats.sigma_clip(data, sigma=sigma)

            new_xvals.extend(domain[~output.mask])
            new_yvals.extend(data[~output.mask])

        new_len = len(new_xvals)

        if (length-new_len) / length >= 0.1:
            print_warning("Over 10% of pixels have been rejected in the sigma_clip routine.")

        assert len(new_xvals) == len(new_yvals)

        return np.array(new_xvals), np.array(new_yvals)


    new_x, new_y = np.array(xvalues), np.array(yvalues)

    if isinstance(sample_size, Iterable):
        for sample in sorted(sample_size, reverse=True):
            new_x, new_y = clip_helper(new_x, new_y, sample, sigma)
        
        return new_x, new_y

    else:
        return clip_helper(xvalues, yvalues, sample_size, sigma)


def sortxy(xvalues, yvalues):
    """
    Since these are coordinates, this function sorts the y values using the x
    values as keys. 
    https://www.geeksforgeeks.org/python-sort-values-first-list-using-second-list/
    """
    assert len(xvalues) == len(yvalues)
    zipped_pairs = zip(xvalues, yvalues)
    sorted_y = [y for _,y in sorted(zipped_pairs)]

    # Correctness check  
    assert len(xvalues) == len(sorted_y)
    
    return sorted(xvalues), sorted_y


def nearest_ind_to_val(arr, val):
    """
    Given some array arr, it returns the index of the element that is closest to
    val. If there are multiple such elements, it returns a list.
    """
    arr = np.array(arr)
    return (np.abs(arr - val)).argmin()


def smooth(x,window_len=11,window='hanning'):
    """smooth the data using a window with requested size.
    
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    
    input:
        x: the input signal 
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal
        
    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)
    
    see also: 
    
    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter
 
    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")


    if window_len<3:
        return x


    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")


    s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')

    y=np.convolve(w/w.sum(),s,mode='valid')
    return y


def fit_parabola(x, y, domain):
    order = 2
    f = construct_polynomial(x, y, order)
    output = f(domain)
    return output


def construct_polynomial(x, y, order):
    polyfit_array = np.polyfit(x, y, order)

    def f(x):
        power = 0
        output = np.zeros(len(x))
        for coef in reversed(polyfit_array):
            output += coef * x**power
            power += 1

        return output

    return f



class ScatterFactory:
    """
    This is a data class to hold the necessary data to plot the spectrum's 
    brightness. We don't want to pollute the Spectrum namespace, so we 
    are doing that in a separate class.
    """
    def __init__(self):
        self.scatter_list = []

    def add_scatter(self, x, y):
        self.scatter_list.append((x, y))

    def scatter(self):
        for x, y in self.scatter_list:
            plt.scatter(x, y)


class PlotFactory:
    """
    This is a data class to hold the necessary data to scatter the spectrum's 
    brightness.
    """
    def __init__(self):
        self.plot_list = []
    
    def add_plot(self, x, y, color=None):
        self.plot_list.append((x, y, color))

    def plot(self):
        for x, y, color in self.plot_list:
            plt.plot(x, y, color=color)


def min_ind_range(array, start, end):
    """
    Obtains the indices of the absolute minima of the first parameter
    in the range provided by the 2nd and 3rd parameter.
    """
    temp = array[start:end]
    min_ind_temp = np.argmin(temp)
    min_ind_real = start + min_ind_temp
    return min_ind_real


def get_vmin_vmax(image):
    fifth = np.percentile(image, 5)
    ninety_fifth = np.percentile(image, 95)

    return fifth, ninety_fifth

def find_int_peaks(intensity_array, height=100, dist=5):
    # ignores peaks with intensities less than 100
    peaks = scipy.signal.find_peaks(intensity_array, height=height, 
                                                        distance=dist) 
    
    return peaks[0]


def find_xy_peaks(x, y):
    """
    Finds peaks in a distribution given the x values and the yvalues. Returns
    the (x, y) coordinates of the peaks.
    """
    peaks,_ = scipy.signal.find_peaks(y)
    xpeaks = np.take(x, peaks)
    ypeaks = np.take(y, peaks)
    return xpeaks, ypeaks


def find_cwt_peaks(x, y, width_array):
    peakind = scipy.signal.find_peaks(y, width_array)
    xpeaks = np.take(x, peakind)
    ypeaks = np.take(y, peakind)
    return xpeaks, ypeaks


def fit_spline(x, y, domain, degree=3):
    """
    Fits a smoothing spline to the x-y values. Then applies to the domain and
    returns an `output` array where the function is applied to `domain`.
    """
    f = scipy.interpolate.UnivariateSpline(x, y, k=degree)
    output = f(domain)
    spline_yvals = f(x)
    rms_val = rms(spline_yvals, y)
    return output, rms_val




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
    fits_file.get_spectra()
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

