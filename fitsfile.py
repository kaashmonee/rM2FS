from astropy.io import fits
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import scipy
from spectrum import Spectrum
import gaussfit
import sys

### global variable for button toggling purposes, modeled off the matplotlib
# documentation here: https://matplotlib.org/2.1.2/gallery/event_handling/keypress_demo.html.


class FitsFile:
    def __init__(self, fits_file):
        """
        Creates a class that opens and contains .fits image attributes. 
        It takes in a fits image path.
        """
        self.fits_file = fits_file
        hdul = fits.open(fits_file)
        self.image_data = hdul[0].data
        self.log_image_data = np.log(self.image_data)
        self.rows = self.image_data.shape[0]
        self.cols = self.image_data.shape[1]
        self.spectra = []
        self.num_spectra = 0

        # self.__get_spectra()


    def get_dimensions(self):
        return (self.rows, self.cols)

    def plot_spectra(self, num_to_plot=None, save=False, show=False):
        """
        Plots the spectra on top of the image.
        """
        # Clear the figure.
        plt.clf()

        if not show and not save:
            raise ValueError("You must either choose to save or show the spectra.")

        if num_to_plot is None: 
            num_to_plot = len(self.spectra)

        # importing inside function to avoid circular dependency issues
        import util

        # Setting up plotting...
        vmin, vmax = util.get_vmin_vmax(self.image_data)
        plt.imshow(self.image_data, origin="lower", cmap="gray", 
                                    vmin=vmin, vmax=vmax)
        
        image_rows = self.image_data.shape[0]
        image_cols = self.image_data.shape[1]
        degree = 3

        spectrum_scatter_plots = []
        fit_plots = []

        for spectrum in self.spectra[:num_to_plot]:
            
            # Uncomment this section if the scatter plot portion of the spectrum
            # is desired.
            spectrum_scatter = spectrum.plot(only_endpoints=True)
            spectrum_scatter_plots.append(spectrum_scatter)

            fit_plot = spectrum.plot_fit()
            fit_plots.append(fit_plot)


        # Plotting...
        plt.plot(self.start_parab, self.parab_range)
        plt.plot(self.end_parab, self.parab_range)

        plt.xlabel("xpixel")
        plt.ylabel("ypixel") 
        plt.title("Image " + self.get_file_name() + " with Spectral Continuum Fits\nSpectra " + str(num_to_plot) + "/" + str(self.num_spectra))
        plt.xlim(0, self.get_dimensions()[1])
        plt.ylim(0, self.get_dimensions()[0])

        current_fig = plt.gcf()

        if save: 
            directory = "completed_images/"
            image_file_name = self.get_file_name() + "_fitted.svg"
            print("Saving " + image_file_name + " to disk...")
            current_fig.savefig(directory + image_file_name, dpi=1500)

        if show: 
            plt.show()




    def __get_intensity_array(self, xpixel=1000):
        """
        At a particular xpixle, (default is 1000), this function returns
        an array of intensity for each pixel in y.
        """
        intensity = []
        for row_num, row in enumerate(self.image_data):
            intensity.append(row[xpixel])

        return np.array(intensity)

    
    def __find_peaks(self):
        """
        Find peaks in the intensity array. The peaks correspond to each order of the spectrograph.
        """
        import util

        prime_pixel = 1000
        length = self.image_data.shape[1]
        xpixels = np.arange(length)

        # Finds number of peaks based on reference x pixel
        intensity_array = self.__get_intensity_array(xpixel=prime_pixel)
        num_peaks_prime = util.find_int_peaks(intensity_array)
        used_xpixels = []

        # The dictionary containing peaks at each x value
        xpeaks = dict()

        # Obtain a dictionary of x pixels and the peaks at each x pixel.
        for xpixel in xpixels:
            ia = self.__get_intensity_array(xpixel=xpixel)
            peaks = util.find_int_peaks(ia)
            xpeaks[xpixel] = peaks

        return xpeaks


    def __plot_peaks(self, xpeaks):
        print("Plotting peaks")
        img = np.zeros((self.image_data.shape[0], self.image_data.shape[1]))
        for x in xpeaks:
            img[xpeaks[x], x] = 1


        plt.imshow(img, origin="lower")
        current_figure = plt.gcf()
        plt.show()

        print("Saving peak plot...") 
        current_figure.savefig("assets/peak_plot.svg", dpi=2000)        


    def get_spectra(self):
        import util

        xpeaks = self.__find_peaks()

        image_length = self.image_data.shape[1]

        start_pixel = 1000
        yvalues_at_start = xpeaks[start_pixel]
        self.num_spectra = len(yvalues_at_start)
        xthreshold = 5
        ythreshold = 2
        cur_num_spectra = 0

        # Going from right to left
        for num, y in enumerate(yvalues_at_start):
            cur_y = y
            s = Spectrum([], [], self)
            
            cur_x = start_pixel
            for next_spec_x in range(start_pixel+1, image_length):

                check_y = xpeaks[next_spec_x]
                # Check for xpixels to see if there exists a y pixel that's less
                # than some value away.
                spec_indices = np.where(abs(cur_y-check_y) <= ythreshold)[0]

                if len(spec_indices) > 0:
                    next_ind = spec_indices[0]
                    nexty = check_y[next_ind]
                    s.add_peak(next_spec_x, nexty)
                    cur_x = next_spec_x
                    cur_y = nexty

                if abs(next_spec_x - cur_x) >= xthreshold:
                    break

            cur_x = start_pixel
            cur_y = y
            for prev_spec_x in range(start_pixel-1, 0, -1):

                check_y = xpeaks[prev_spec_x]
                spec_indices = np.where(abs(cur_y-check_y) <= ythreshold)[0]

                if len(spec_indices) > 0:
                    prev_ind = spec_indices[0]
                    prevy = check_y[prev_ind]
                    s.add_peak(prev_spec_x, prevy)
                    cur_x = prev_spec_x
                    cur_y = prevy

                if abs(prev_spec_x - cur_x) >= xthreshold:
                    break

            build_prep_success = s.build_prepare()
            if build_prep_success:
                cur_num_spectra += 1
                self.spectra.append(s)
                print("Spectrum %d/%d ready for building..." % (cur_num_spectra, self.num_spectra))
            else:
                self.num_spectra -= 1


        self.__fit_overlap_boundary_parabola()
        self.__update_spectral_boundaries()
        
        built_spectra = []
        cur_num_spectra = 0

        for spectrum in self.spectra: 

            build_success = spectrum.build()

            if build_success: 
                cur_num_spectra += 1
                built_spectra.append(spectrum)
                print("Building spectrum %d/%d" % (cur_num_spectra, self.num_spectra))
                print("Min x:", s.xvalues[0], "\nMax x:", s.xvalues[-1])
            else:
                self.num_spectra -= 1

        self.spectra = built_spectra


    def plot_spectra_brightness(self):
        for ind, spectrum in enumerate(self.spectra):
            num = ind + 1 
            spectrum.plot_spectrum_brightness(num)

        # plt.show()


    def __fit_overlap_boundary_parabola(self):
        """
        This function fits a parabola to the overlap boundaries after the 
        spectrum.remove_overlap_spectrum is run. However, since a vertical 
        parabola needs to be fit, the y value are the effective xvalues and 
        the xvalues are the effective yvalues.
        """
        import util

        spectrum_startx = []
        spectrum_starty = []
        spectrum_endx = []
        spectrum_endy = []

        for spectrum in self.spectra:
            spectrum_startx.append(spectrum.int_xvalues[0])
            spectrum_starty.append(spectrum.int_yvalues[0])
            spectrum_endx.append(spectrum.int_xvalues[-1])
            spectrum_endy.append(spectrum.int_yvalues[-1])

        height = self.image_data.shape[0]
        domain = np.arange(height)

        start_parab, sp_rms = util.fit_parabola(spectrum_starty, 
                                                spectrum_startx, 
                                                domain)

        end_parab, ep_rms = util.fit_parabola(spectrum_endy, spectrum_endx, 
                                              domain)

        self.start_parab = start_parab
        self.end_parab = end_parab
        self.parab_range = domain

        # StartParabola_rms and EndParabola_rms
        self.sp_rms = sp_rms
        self.ep_rms = ep_rms


    def __update_spectral_boundaries(self):
        """
        Fixes the spectral boundaries by replacing the spectral boundaries with 
        points that should be closer to the edges.
        """
        import util

        height = self.image_data.shape[0]
        domain = np.arange(height)

        print("spectral boundaries before...")

        for spectrum in self.spectra:
            # Updating the left half boundaries
            startx = spectrum.int_xvalues[0]
            starty = spectrum.int_yvalues[0]
            
            parab_starty_ind = util.nearest_ind_to_val(domain, starty)
            start_parabx = self.start_parab[parab_starty_ind]
            
            # If the true yvalue is greater than 3 std. dev. away from the 
            # parabola, replace the y value with the x value at the parabola
            int_xvals, int_yvals = [], []

            # Starting and ending indices of the first and last 
            # values of the int_xvalues array in the spectrum.ox array
            spec_start_ind = spectrum.ox.index(spectrum.int_xvalues[0])
            spec_end_ind = spectrum.ox.index(spectrum.int_xvalues[-1])

            if abs(start_parabx - startx) >= 3 * self.sp_rms:
                # Obtain index of xvalue that is closest to the starting value
                # of the parabola
                print("start_parabx:", start_parabx, "startx:", startx)
                spec_start_ind = util.nearest_ind_to_val(spectrum.ox, start_parabx)


            # Repeat same procedure as above for the last values
            endx = spectrum.int_xvalues[-1]
            endy = spectrum.int_yvalues[-1]

            parab_endy_ind = util.nearest_ind_to_val(domain, endy)
            end_parabx = self.end_parab[parab_endy_ind]
            
            if abs(end_parabx - endx) >= 3 * self.ep_rms:
                print("end_parabx:", end_parabx, "endx:", endx)
                spec_end_ind = util.nearest_ind_to_val(spectrum.ox, end_parabx)
            
            int_xvals = spectrum.ox[spec_start_ind:spec_end_ind+1]
            int_yvals = spectrum.oy[spec_start_ind:spec_end_ind+1]

            # Update the spectrum int_xvals and the int_yvals
            spectrum.int_xvalues = int_xvals
            spectrum.int_yvalues = int_yvals




    def get_file_name(self):
        """
        Returns the name of the file independent of the path.
        """
        return self.fits_file[self.fits_file.rfind("/")+1:]

