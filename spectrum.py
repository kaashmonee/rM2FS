import numpy as np
import matplotlib.pyplot as plt
import scipy
from astropy import modeling
import gaussfit
from gaussfit import Peak


class Spectrum:
    """
    This is a class that represents each white spot on the image.
    """
    spectrum_number = 1

    def __init__(self, xvalues, yvalues, fits_file):
        import util
        # These variables must be lists so that we can add spectra. They will
        # be turned into numpy arrays after we call the `build` function.
        self.xvalues = list(xvalues)
        self.yvalues = list(yvalues)
        self.fits_file = fits_file
        self.image_rows = self.fits_file.image_data[0]
        self.image_cols = self.fits_file.image_data[1]

        # Peaks for which a Gaussian was unable to be fit
        self.unfittable_peaks = []

        # In order to establish a width profile, each fitted Gaussian also 
        # contains a standard deviation/width value that is stored and fitted
        # a spline.
        self.peak_width_spline_function = None
        self.peak_width_spline_rms = None
        self.widths = None

        # When plotting the brightness of the spectrum, we are going to need 
        # to store relevant information, which will be in the following varaible
        # This should be of type SpectrumBrightnessPlotData and will be updated
        # in the __remove_overlapping_spectra method.
        self.spectrum_brightness_plot_data = None

        Spectrum.spectrum_number += 1

    
    def add_peak(self, x, y):
        """
        Adds a peak to the spectrum.
        """
        self.xvalues.append(x)
        self.yvalues.append(y)



    def build(self):
        import util
        """
        After all the points have been added to the spectrum, this function
        must be called to 'build' the spectrum, which performs Gaussian fits of 
        the integer peaks, removes outliers, removes the overlapping portions
        of the spectra, and establishes a width profile. If there are fewer than
        100 points in the spectrum, then the build is rejected and False
        is returned.
        """

        # Sorting the x and y values
        self.xvalues, self.yvalues = util.sortxy(self.xvalues, self.yvalues)

        # Ensuring that we keep track of the integer yvalues
        # This is useful for when we want to plot the brightness vs. x value of
        # the peaks.
        self.int_xvalues = np.array(self.xvalues)
        self.int_yvalues = np.array(self.yvalues)
        assert len(self.int_xvalues) == len(self.int_yvalues)

        # Run the remove overlapping spectra method, which will update the 
        # self.int_xvalues and self.int_yvalues variables. We will use those
        # to update the self.xvalues and self.yvalues variables.
        self.__remove_overlapping_spectrum()

        self.xvalues, self.yvalues = self.int_xvalues, self.int_yvalues

        # Ensuring that the spectrum has a reasonable size...
        xlen = len(self.xvalues)
        ylen = len(self.yvalues)
        if xlen <= 100:
            print("Build rejected! Fewer than 100 points in the spectrum...")
            return False

        if np.diff(self.xvalues).all() <= 0:
            print("self.xvalues:", self.xvalues)
            plt.plot(self.xvalues)
            plt.show()

        # Adding a correctness check to ensure that the dimensions of each are correct.
        if xlen != ylen:
            raise ValueError("The dimensions of the xvalues and yvalues array are not the same; xlen:", xlen, " ylen:", ylen)

        # Generating peaks after the spectrum is cleaned and narrowed
        self.peaks = [Peak(x, y) for x, y in zip(self.xvalues, self.yvalues)]

        # Fits a Gaussian to each of the peaks.
        self.__fit_peak_gaussians()

        # Removes outliers
        self.__remove_outliers()

        # After obtaining the true y values and narrowing the spectrum,
        # we want to fit the spectrum with a UnivariateSpline, which is 
        # what this function does.
        degree = 3
        self.__fit_spectrum(np.arange(0, len(self.image_cols)), degree)

        # This function fits a spline to the peak widths and generates an rms 
        # value.
        self.__fit_peak_widths()

        return True


    def __fit_peak_gaussians(self):
        """
        This function fits a Gaussian to each spectrum in the .fitsfile. Then, 
        it finds the center of the Gaussian and assigns it to the peak.true_center
        parameter of each Peak object in each Spectrum. 
        """
        image = self.fits_file.image_data
        image_height = image.shape[1]

        print("Fitting gaussian...")
        import time
        t1 = time.time()
        success_counter = 0

        for peak in self.peaks:
            y1 = peak.y
            
            # This is just an arbitrary range that has been chosen.
            # This might have to be tweaked for various spectra.
            left_range = 4
            right_range = 4
            y0 = y1 - left_range
            y2 = y1 + right_range

            # Ensure that the ranges do not exceed the width and height of 
            # the image
            if y0 <= 0: y0 = 0
            if y2 >= image_height: y2 = image_height
            rng = (y0, y2)

            # This does the fitting and the peak.true_center setting.
            show = False
            success = gaussfit.fit_gaussian(self.fits_file, rng, peak,
                                            show=show)
            success_counter += success

        if success_counter != len(self.peaks):
            print("\nSpectrum %d had %d/%d points with a successful Gaussian fit." % (Spectrum.spectrum_number, success_counter, len(self.peaks)))


        self.yvalues = np.array([peak.true_center for peak in self.peaks])

        #############################################################
        # This is for testing purposes only --- this code should be #
        # deleted after testing is complete because no plotting sh- #
        # ould be taking place here.                                # 
        #############################################################

        # if Spectrum.spectrum_number in [15, 21]:
        #     self.fits_file.plot_spectra(num_to_plot=Spectrum.spectrum_number, 
        #                                 show=True, save=False) 
            # spectrum.plot_peak_widths()

        # if spec_ind == len(self.spectra):
        #     self.plot_spectra(num_to_plot=spec_ind, save=True)




    def plot(self, only_endpoints=True):
        """
        Takes in an optional parameter `show` that shows the plot as well.
        """
        size = 0.75

        xvalues_to_plot = self.xvalues
        yvalues_to_plot = self.yvalues

        if only_endpoints:
            xvalues_to_plot = [xvalues_to_plot[0], xvalues_to_plot[-1]]
            yvalues_to_plot = [yvalues_to_plot[0], yvalues_to_plot[-1]]

        scatter_plot = plt.scatter(xvalues_to_plot, yvalues_to_plot, s=size)

        
        return scatter_plot


    def __fit_spectrum(self, domain, degree):
        """
        This function fits a polynomial of degree `degree` and saves
        the output on the input domain. It then saves the RMS goodness of fit
        value.
        """
        import util
        self.output, self.rms_value = util.fit_spline(self.xvalues, self.yvalues,
                                                      domain, degree=degree)


    def plot_spectrum_brightness(self, num):
        """
        Plots the brightness of each spectra against the xvalue.
        """
        import util
        plt.clf()
        dat = self.spectrum_brightness_plot_data

        plt.scatter(dat.brightness_xvals_to_plot, dat.brightness_array)
        plt.scatter(self.int_xvalues, self.fits_file.image_data[self.int_yvalues, self.int_xvalues])
        plt.plot(dat.brightness_xvals_to_plot, dat.smoothed_brightness, color="red")
        plt.scatter(dat.extremax, dat.extremabright)

        image_name = self.fits_file.get_file_name()
        plt.title("brightness vs. xvalues in %s, spectrum #: %d" % (image_name, num))
        plt.xlabel("xpixel")
        plt.ylabel("brightness")
        
        plt.show()


    def plot_fit(self):
        linewidth = 0.25

        fit_plot = plt.plot(self.output, linewidth=linewidth)
        return fit_plot

    def __remove_outliers(self):
        import util
        sample_sizes = [10, 20, 50, 70, 100, 300]
        self.xvalues, self.yvalues = util.sigma_clip(
            self.xvalues, self.yvalues, sample_size=sample_sizes, sigma=3
        )

    
    def __remove_overlapping_spectrum(self):
        """
        This routine removes the part of the spectrum on either side that is 
        responsible that overlaps. This way, we only use the middle spectrum for 
        our analysis. This routine works as follows:
        1. It obtains the brightness vs. x plots for each spectrum. 
        2. It then uses a Savitzky-Golay filter to smooth the scatter. 
        3. It obtains the local minima in the smoothed scatter. After finding 
        the local minima, there are 3 things on which to case:
            a. Greater than 2 local minima
                If there are more than 2 local minima, then remove the minima
                closest to the edges until there are only 2 remaining.
            b. Two minima detected
                Ensure that the two minima are not in the same half of image.
                If they are, then remove the minimum closest to the edge.
            c. Fewer than two minima detected
                Do nothing
        """

        import util

        # Obtaining an array of brightness values for each spectrum
        brightness_array = self.fits_file.image_data[self.int_yvalues, self.int_xvalues]

        # This dictionary is for restoring the yvalues after the xvalues with
        # pixels that are too bright are clipped out
        brightness_dict = {x: y for (x, y) in zip(self.int_xvalues, self.int_yvalues)}

        # Sigma clipping the brightness array to get rid of the extreme values
        # Ensures that the next line restores the yvalues and that the x and y
        # correspond
        clip_window = 100
        self.int_xvalues, brightness_array = util.sigma_clip(self.int_xvalues,
                                            brightness_array, 
                                            sample_size=clip_window)
        self.int_yvalues = [brightness_dict[x] for x in self.int_xvalues]

        # xvalues that will be used for plotting
        to_plot_x = np.array(self.int_xvalues)

        # Correctness check
        assert len(self.int_xvalues) == len(self.int_yvalues)
        assert len(self.int_xvalues) == len(brightness_array)

        # Smoothing the brightness array
        # Obtain window size
        window_size = len(self.int_xvalues) // 8

        if window_size % 2 == 0:
            window_size -= 1
        
        order = 3
        smoothed_brightness = scipy.signal.savgol_filter(brightness_array, window_size, order)

        # Obtaining the minima of the smoothed function and the x indices of the
        # minima
        extrema_indices = scipy.signal.argrelextrema(smoothed_brightness, np.less, order=100)

        # Obtaining the minima
        extremax = self.int_xvalues[extrema_indices]
        extremabright = smoothed_brightness[extrema_indices] 

        # Case on the 3 different possible scenarios as described in the
        # docstring...

        # Correctness check
        assert len(extremax) == len(extremabright)

        image_width = len(self.image_cols)

        # If there are greater than 2 minima, keep removing the ones closest
        # to the edges until there are exactly 2 left
        extremax, extremabright = util.sortxy(extremax, extremabright)
        if len(extremax) > 2:
            while len(extremax) != 2:
                if extremax[0] <= abs(extremax[-1] - image_width):
                    extremax.pop(0)
                    extremabright.pop(0)
                else:
                    extremax.pop()
                    extremabright.pop()

        # Just another correctness check
        assert len(extremax) == len(extremabright)

        # If there are exactly 2 maxima
        halfway_point = (self.int_xvalues[-1] - self.int_xvalues[0]) // 2
        if len(extremax) == 2:

            x1 = extremax[0]
            x2 = extremax[1]

            if x1 <= halfway_point and x2 <= halfway_point:
                extremax.pop(0)
                extremabright.pop(0)
            elif x1 >= halfway_point and x2 >= halfway_point:
                extremax.pop()
                extremabright.pop()


            # The minima points should now represent the starting and ending
            # points of the spectra
            assert len(extremax) == len(extremabright)
            assert len(extremax) <= 2
            assert len(self.int_xvalues) == len(self.int_yvalues)

            # If no points were removed above
            if len(extremax) == 2:
                startx = list(self.int_xvalues).index(extremax[0])
                endx = list(self.int_xvalues).index(extremax[1])
                self.int_xvalues = self.int_xvalues[startx:endx+1]
                self.int_yvalues = self.int_yvalues[startx:endx+1]

            assert len(self.int_xvalues) == len(self.int_yvalues)


        if len(extremax) == 1:

            # If the point is on the left side of the image, take the values 
            # from the point to the end of the image
            assert len(self.int_xvalues) == len(self.int_yvalues)
            if extremax[0] < halfway_point:
                startx = list(self.int_xvalues).index(extremax[0])
                self.int_xvalues = self.int_xvalues[startx:]
                self.int_yvalues = self.int_yvalues[startx:]
                assert len(self.int_xvalues) == len(self.int_yvalues)

            # If the point is on the right, then take the values from the point
            # to the left of the image
            elif extremax[0] > halfway_point:
                endx = list(self.int_xvalues).index(extremax[0])
                self.int_xvalues = self.int_xvalues[:endx]
                self.int_yvalues = self.int_yvalues[:endx]
                assert len(self.int_xvalues) == len(self.int_yvalues)
            
            else:
                raise ValueError("The extrema value should not be in the middle")

            assert len(self.int_xvalues) == len(self.int_yvalues)

        assert len(self.int_xvalues) == len(self.int_yvalues)        

        # Setting instance variables so that they can be used in the plotting
        # function
        self.spectrum_brightness_plot_data = SpectrumBrightnessPlotData(
            smoothed_brightness, 
            brightness_array, 
            to_plot_x, 
            extremax, 
            extremabright
        )


    
    def __fit_peak_widths(self):
        """
        For each peak, fits a spline to a plot to the function of the standard
        deviation fitted Gaussian to the x value.
        """
        import util
        # self.widths = np.array([peak.width for peak in self.peaks])
        self.widths = []
        xvalues = []

        # Ensuring that the peaks with unfittable Gaussians won't be included
        for peak in self.peaks:
            if peak.width == "failed":
                self.unfittable_peaks.append(peak)
                continue
            self.widths.append(peak.width)
            xvalues.append(peak.x)

        self.widths = np.array(self.widths)
        xvalues = np.array(xvalues)

        # TODO: This should be replaced with the more abstract util.fit_spline
        # function --- that is the function that should be used for all spline
        # fitting in this codebase. This is addressed in #93.
        f = scipy.interpolate.UnivariateSpline(xvalues, self.widths)
        widths_spline = f(xvalues)
        self.peak_width_spline_function = f
        self.peak_width_spline_rms = util.rms(widths_spline, self.widths)


    def plot_peak_widths(self):
        """
        Plotting function to plot the peak widths. This can only be called after
        gaussfit.fit_gaussian is called. It then fits a univariate spline to it. 
        The fit_peak_widths function must be called in order for this function
        to run.
        """
        xvalues = self.xvalues

        # Safety check to ensure that the user fits the peak widths before 
        # trying to plot them.
        if self.widths is None:
            raise RuntimeError("The plot_peak_widths function was called before the fit_peak_widths function was called.")

        widths = self.widths
        widths_spline = self.peak_width_spline_function(xvalues)

        plt.scatter(xvalues, widths)
        plt.plot(xvalues, widths_spline, color="red")
        plt.xlabel("xpixel")
        plt.ylabel("width")
        plt.title("gaussian width v. peak")
        
        # Adding the rms of the spline fit to the plot.
        ax = plt.gca()
        rms_text = "rms: " + str(self.peak_width_spline_rms)
        ax.text(5, 5, rms_text)

        plt.show()


class SpectrumBrightnessPlotData:
    """
    This is a data class to hold the necessary data to plot the spectrum's 
    brightness. We don't want to pollute the Spectrum namespace, so we 
    are doing that in a separate class.
    """
    def __init__(self, smoothed_brightness, brightness_array, brightness_xvals, extremax, extremabright):
        self.smoothed_brightness = smoothed_brightness
        self.brightness_array = brightness_array
        self.brightness_xvals_to_plot = brightness_xvals
        self.extremax = extremax
        self.extremabright = extremabright

