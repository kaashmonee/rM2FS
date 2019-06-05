import unittest
import numpy as np
from continuum import Spectrum

class SpectrumUnitTest(unittest.TestCase):
    """
    Unit tests for the Spectrum class.
    """

    def test_narrow(self):
        """
        Unit testing the narrow function.
        """
        xvals = np.array([1, 2, 3, 4, 5])
        yvals = np.array([2, 2.5, 4, 8, 9])
        self.test_spectrum = Spectrum(xvals, yvals)
        print(self.test_spectrum.xvalues)
        print(self.test_spectrum.yvalues)
        self.assertTrue(self.test_spectrum.xvalues.all() == np.array([1, 2]).all())
        self.assertTrue(self.test_spectrum.yvalues.all() == np.array([2, 2.5]).all())


if __name__ == "__main__":
    unittest.main()

