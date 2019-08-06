import numpy as np
import matplotlib.pyplot as plt
import scipy
from astropy import modeling
import gaussfit
from gaussfit import Peak
from util import PlotFactory, ScatterFactory


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
        self.spec_scatter_fact = ScatterFactory()
        self.spec_plot_fact = PlotFactory()

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

        # Ensuring no duplicates and ensuring strictly increasing
        # assert np.diff(self.xvalues).all() <= 0

        # Adding a correctness check to ensure that the dimensions of each are correct.
        if xlen != ylen:
            raise ValueError("The dimensions of the xvalues and yvalues array are not the same; xlen: %d ylen: %d" % (xlen, ylen))

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

        if Spectrum.spectrum_number in [3, 7, 21]:
            self.fits_file.plot_spectra(num_to_plot=Spectrum.spectrum_number, 
                                        show=True, save=False)
            self.fits_file.plot_spectra_brightness() 

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
        self.spec_scatter_fact.scatter()
        self.spec_plot_fact.plot()

        image_name = self.fits_file.get_file_name()
        plt.title("-parabola/brightness vs. xvalues in %s, spectrum #: %d" % (image_name, num))
        plt.xlabel("xpixel")
        plt.ylabel("-parabola/brightness")
        
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
        3. It obtains the local maxima in the smoothed scatter. After finding 
        the local maxima, a parabola should be fit. The parabola will then be
        divided by the brightness plot. The absolute minima in the resulting 
        plot can be used as the starting and ending points of the overlapping 
        spectra. There are 3 things on which to case.
            a. One local maximum:
                If there is only 1 local maximum, there is no need to cut the 
                spectrum. The spectrum starts and ends have been detected 
                appropriately.
            b. 2 local maxima:
                If there are 2 local maxima, there are 2 cases. If the 2 local
                maxima are on the left side of the image, pick a point on the 
                right side of the brightness plot for which to fit the parabola.
                Do the same on the opposite side if the maxima are on the 
                opposite half.
            c. 3 local maxima:
                If there are 3 local maxima, simply fit a parabola to all 3.
        """

        import util

        # Obtaining an array of brightness values for each spectrum
        brightness_array = self.fits_file.image_data[self.int_yvalues, self.int_xvalues]

        # This dictionary is for restoring the yvalues after the xvalues with
        # pixels that are too bright are clipped out
        brightness_dict = {x: y for (x, y) in zip(self.int_xvalues, self.int_yvalues)}

        # Sigma clipping the brightness array to get rid of the extreme values
        # Ensures that the next line restores the yvalues and that the x and y
        # correspond. This clips away pixels that are much brigher than expected
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
        # Obtain window size --- a larger window size is associated with
        # a smoother output
        window_size = len(self.int_xvalues) // 6

        if window_size % 2 == 0:
            window_size -= 1
        
        order = 3
        smoothed_brightness = scipy.signal.savgol_filter(brightness_array, 
                                                         window_size, order)
        
        # Correctness check
        assert len(self.int_xvalues) == len(self.int_yvalues)
        assert len(self.int_xvalues) == len(brightness_array)

        # Obtaining the minima of the smoothed function and the x indices of the
        # minima
        order = 100
        max_extrema_indices = scipy.signal.argrelextrema(smoothed_brightness, 
                                                         np.greater, 
                                                         order=order)

        # Obtaining the minima
        max_extremax = self.int_xvalues[max_extrema_indices]
        max_extrema = smoothed_brightness[max_extrema_indices]

        # Additional correctness checks
        assert len(smoothed_brightness) == len(self.int_xvalues)

        # Correctness check
        image_width = len(self.image_cols)

        # If there are greater than 2 minima, keep removing the ones closest
        # to the edges until there are exactly 2 left
        max_extremax, max_extrema = util.sortxy(max_extremax, max_extrema)

        # Correctness check
        assert len(max_extremax) == len(max_extrema)

        num_max = len(max_extremax)

        # If num_max > 3, then we've got cases that we haven't accounted for
        assert num_max <= 3

        if num_max == 3:
            # Fits a parabola
            py = util.fit_parabola(max_extremax, max_extrema, self.int_xvalues)

            divided_plot = -py / smoothed_brightness

            # Finds the absolute minima in the first and second halves of the 
            # image
            length = len(self.int_xvalues)

            ## Some correctness checkds before proceeding
            assert length == len(self.int_xvalues)
            assert length == len(divided_plot)

            # Obtains the indices of the absolute minima of the first parameter
            # in the range provided by the 2nd and 3rd parameter
            min_left_ind = util.min_ind_range(divided_plot, 0, length//2)
            min_right_ind = util.min_ind_range(divided_plot, length//2, length)

            assert len(self.int_xvalues) == len(self.int_yvalues)
            cleanedx = self.int_xvalues[min_left_ind:min_right_ind]
            cleanedy = self.int_yvalues[min_left_ind:min_right_ind]
            cleaned_brightness = brightness_array[min_left_ind:min_right_ind]

            # self.spec_plot_fact.add_plot(self.int_xvalues, divided_plot)
            self.spec_scatter_fact.add_scatter(self.int_xvalues, brightness_array)
            self.spec_scatter_fact.add_scatter(cleanedx, cleaned_brightness)
        
        elif num_max == 2:
            pass

        elif num_max == 1:
            pass

        else: # num_max must be between 1 and 3
            raise ValueError("num_max = %d. This case is not accounted for." % (num_max))



        # Adding a list of things to plot...
        # self.spec_scatter_fact.add_scatter(to_plot_x, brightness_array)
        # self.spec_scatter_fact.add_scatter(max_extremax, max_extrema)
        # self.spec_plot_fact.add_plot(to_plot_x, smoothed_brightness,color="red")


    
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



