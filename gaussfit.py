# Routines for gauss fitting.
import numpy as np
import scipy
from astropy import modeling
import matplotlib.pyplot as plt
import math
import warnings
import scipy.stats

class Peak:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.true_center = None
        self.width = None # the sigma value of the fitted Gaussian
        self.anderson_test = None
        self.dagostino_test = None
        self.shapiro_test = None


def gauss(x, amp, cen, wid):
    """
    Gauss fitting function. Using the precision (tau) to define the width of the
    distribution, as mentioned here: 
    https://en.wikipedia.org/wiki/Normal_distribution.
    """
    return (amp / ((2*math.pi)**0.5 * wid)) * scipy.exp(-(x-cen)**2 / (2*wid**2))


def is_data_gaussian(data, peak):
    """
    This function determines if the data is Gaussian using the Shapiro-Wilk test,
    D'Agostino's K^2 test, and the Anderson-Darling test, as suggested in 
    https://machinelearningmastery.com/a-gentle-introduction-to-normality-tests-in-python/.
    """
    shapiro_test, dagostino_test, anderson_test = True, True, True
    alpha = 0.05

    # Shapiro-Wilk test
    shapiro_stat, shapiro_p = scipy.stats.shapiro(data)

    if shapiro_p <= alpha:
        # print("Data point not Gaussian, as determined by Shapiro-Wilk test.")
        shapiro_test = False

    # Dagostino's K^2 test
    dagostino_stat, dagostino_p = scipy.stats.normaltest(data)
    if dagostino_p <= alpha:
        # print("Data not Gaussian, as determined by Shapiro-Wilk test.")
        dagostino_test = False

    # Anderson-Darling test
    anderson_result = scipy.stats.anderson(data)
    reject_H0_list = []
    for i in range(len(anderson_result.critical_values)):
        sl, cv = anderson_result.significance_level[i], anderson_result.critical_values[i]
        if anderson_result.statistic < anderson_result.critical_values[i]:
            reject_H0_list.append(False)
        else:
            reject_H0_list.append(True)

    anderson_test = reject_H0_list[2] # this is the p = 0.05 value

    # Encompassing the nature of the test in the peak value
    peak.anderson = anderson_test
    peak.dagostino = dagostino_test
    peak.shapiro = shapiro_test

    # All tests fail
    if not dagostino_test and not shapiro_test and not anderson_test:
        return "hard_fail"

    # All tests succeed
    elif dagostino_test and shapiro_test and anderson_test:
        return "success"

    # Some tests fail and some tests succeed
    else:
        return "soft_fail"



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

    # Determing if the intensity array is Gaussian. If it is not, then there is 
    # no reason to do a Gaussian fit, so we will just not modify the peak 
    # object.
    # if is_data_gaussian(intensity, peak) != "success":
    #     peak.true_center = "failed"
    #     return

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
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        popt, pcov = scipy.optimize.curve_fit(gauss, x, y, 
                                            p0=[peak_value, mean, sigma], 
                                            maxfev=10000)


    mean_intensity = popt[0]
    mean_y = popt[1]
    peak.true_center = mean_y
    peak.width = popt[2]
    

    fit = gauss(x_continuous, *popt)

    if show:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(yrange, intensity, color="red")
        ax.plot(x_continuous, fit, label="fit")

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

