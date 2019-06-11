from astropy.io import fits
import matplotlib
import numpy as np
import scipy.signal
import os
import argparse
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import cv2
import time
from fitsfile import FitsFile
from numpy.polynomial.legendre import legfit
from numpy.polynomial.legendre import Legendre
from matplotlib.widgets import Button
import cleanup
import argparse
from spectrum import Spectrum

class GaussFit:
    def __init__(self, spectrum):
        self.spectrum = spectrum

    def fit_guassian(self):
        pass


def export_spectra(file_name, spectra):
    """
    Exports the fit polynomials. This can be run only after 
    Spectrum.fit_polynomial is run.
    """
    polynomials = np.array([spectrum.poly for spectrum in spectra])
    np.savetxt(file_name, polynomials, delimiter=",")

def perform_fits(fits_file):
    fits_file.get_spectra()
    fits_file.plot_spectra(show=True)

